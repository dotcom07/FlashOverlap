import torch
import json
from pathlib import Path
import pandas as pd
import argparse
import os
import time
import torch.distributed as dist

torch.ops.load_library("../build/lib/libst_pybinding.so")

WARM_UP = 20
REP = 200

def div_up(x: int, y: int):
    return (x + y - 1) // y

def reorder_indices(S, hint):
    original = list(range(S))
    new_order = [-1] * S
    for i, element in enumerate(hint):
        new_order[element] = i
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    return torch.tensor(new_order, dtype=torch.int, device="cuda")

def generate_row_remap_array(M, N, BM, BN, S_list, world_size, device="cuda"):
    total_tiles = (M * N) // (BM * BN)
    assert sum(S_list) == total_tiles, "sum(S_list) must equal total number of tiles"
    
    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)
    
    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row : current_row + chunk_size]
        mod_values = chunk_row_ids % world_size
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]
        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size
    
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    return remap
    
def init_distributed():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        world_size=world_size,
        rank=rank
        )
    return rank, local_rank, world_size

def generate_and_broadcast_nccl_id():
    rank = dist.get_rank()
    
    if rank == 0:
        nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
        nccl_id_tsr = torch.tensor(nccl_id, device="cuda")
        dist.broadcast(nccl_id_tsr, src=0)
    else:
        nccl_id_tsr = torch.zeros(16, dtype=torch.int64, device="cuda")
        dist.broadcast(nccl_id_tsr, src=0)
        nccl_id = nccl_id_tsr.cpu().tolist()
    
    return nccl_id

def perf_running(M: int, N: int, K: int, BM: int, BN: int, Algo: int, 
               cSeg: list, hint: list, comm_op: str):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    
    nccl_id = generate_and_broadcast_nccl_id()
    
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    gemm_class.nccl_init(rank, world_size, nccl_id)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()
    
    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32) 
    cSeg_GPU = cSeg_CPU.cuda(local_rank)
    
    TileNum = div_up(M, BM) * div_up(N, BN)
    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    
    MonitoredMatrix = torch.zeros(((N+BN-1)//BN), dtype=torch.int, device="cuda")
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M+BM-1)//BM, (N+BN-1)//BN))
    
    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)
    
    _warm_up = WARM_UP
    _freq = REP
    
    if len(cSeg) == 1:
        # No overlapping
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                gemm_class.gemm_allreduce(A, B, C, Algo)

            gemm_class.gemm_allreduce(A, B, C, Algo)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_allreduce(A, B, C, Algo)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter(A, B, C, D, Algo)

            MonitoredMatrix[0] = 0
            gemm_class.gemm_reducescatter(A, B, C, D, Algo)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_reducescatter(A, B, C, D, Algo)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    else:
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        else:
            dur = torch.zeros((_freq))
    
    local_time = torch.tensor([torch.mean(dur).item()], device='cuda')
    dist.all_reduce(local_time, op=dist.ReduceOp.MAX)
    
    return local_time.item()

def perf_comm(M: int, N: int, comm_type: str):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    
    nccl_id = generate_and_broadcast_nccl_id()
    
    comm_class = torch.classes.flashoverlap_class.OverlapImpl()
    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()
    
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    if comm_type == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")

    if comm_type == "all_reduce":
        for _ in range(WARM_UP):
            comm_class.nccl_allreduce(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.nccl_allreduce(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_type == "reduce_scatter":
        for _ in range(WARM_UP):
            comm_class.nccl_reducescatter(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.nccl_reducescatter(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))
        
    local_time = torch.tensor([torch.mean(dur).item()], device='cuda')
    dist.all_reduce(local_time, op=dist.ReduceOp.MAX)
    
    return local_time.item()

def perf_baseline(M: int, N: int, K: int, comm_op: str):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    
    nccl_id = generate_and_broadcast_nccl_id()
    
    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    
    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
    
    gemm_comm = torch.classes.flashoverlap_class.BaselineImpl()
    gemm_comm.nccl_init(rank, world_size, nccl_id)
    gemm_comm.cublas_init()
    
    if comm_op == "all_reduce":
        for _ in range(WARM_UP):
            gemm_comm.gemm_allreduce(A, B, C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            # torch.cuda.cudart().cudaProfilerStart()
            gemm_comm.gemm_allreduce(A, B, C)
            # torch.cuda.cudart().cudaProfilerStop()
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_op == "reduce_scatter":
        for _ in range(WARM_UP):
            gemm_comm.gemm_reducescatter(A, B, C, D)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            # torch.cuda.cudart().cudaProfilerStart()
            gemm_comm.gemm_reducescatter(A, B, C, D)
            # torch.cuda.cudart().cudaProfilerStop()
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))
    
    local_time = torch.tensor([torch.mean(dur).item()], device='cuda')
    dist.all_reduce(local_time, op=dist.ReduceOp.MAX)
    
    return local_time.item()

def main(args):
    rank, local_rank, world_size = init_distributed()
    
    rank = int(os.environ.get('RANK', 0))
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    wave_size = sm_count - 2

    comm_op = args.comm_op
    m, n, k = args.m_dim, args.n_dim, args.k_dim

    file_path = f'../configs/m{m}n{n}k{k}_{gpu_name}.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tile_num = m // data["BM"] * n // data["BN"]
    wave_num = (tile_num + wave_size - 1) // wave_size

    gemm_dur = data["dur"]
    comm_dur = perf_comm(m, n, comm_op)
    overlap_dur = perf_running(m, n, k, data["BM"], data["BN"], 
                             data["Algo"], data["cSeg"], data["hint"], comm_op)
    baseline_dur = perf_baseline(m, n, k, comm_op)

    if rank == 0:
        speedup = baseline_dur / overlap_dur if overlap_dur > 0 else 0
        print(f"""
            {'Item':<10} {'Value':>15}
            {'-----':<10} {'-----':>15}
            {'m':<10} {m:>15}
            {'n':<10} {n:>15}
            {'k':<10} {k:>15}
            {'tile_num':<10} {tile_num:>15}
            {'gemm_dur (ms)':<10} {gemm_dur:>15.4f}
            {'comm_dur (ms)':<10} {comm_dur:>15.4f}
            {'baseline_dur (ms)':<10} {baseline_dur:>15.4f}
            {'overlap_dur (ms)':<10} {overlap_dur:>15.4f}
            {'speedup':<10} {speedup:>15.4f}
            """)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--m_dim', type=int, default=4096)
    parser.add_argument('--k_dim', type=int, default=16384)
    parser.add_argument('--n_dim', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    args = parser.parse_args()

    main(args)
