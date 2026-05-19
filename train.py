import numpy as np
import numba
from numba import njit, prange
import json
import time
import os

# 16^6 = 16777216 possible states for a 6-tile tuple
TABLE_SIZE = 16777216

# We will maintain 3 isomorphic tables.
# 0: 2x3 Rectangles
# 1: L-shape 1
# 2: L-shape 2
NUM_TABLES = 3
alpha = 0.0025 # Learning rate

@njit(nogil=True)
def pack_6(a, b, c, d, e, f):
    return (a << 20) | (b << 16) | (c << 12) | (d << 8) | (e << 4) | f

def get_sym_indices():
    # 0 to 15 mapped to 4x4
    base = np.arange(16, dtype=np.int32).reshape(4, 4)
    syms = np.zeros((8, 4, 4), dtype=np.int32)
    syms[0] = base
    syms[1] = np.rot90(base, 1)
    syms[2] = np.rot90(base, 2)
    syms[3] = np.rot90(base, 3)
    syms[4] = np.fliplr(syms[0])
    syms[5] = np.fliplr(syms[1])
    syms[6] = np.fliplr(syms[2])
    syms[7] = np.fliplr(syms[3])
    
    rect_idx = []
    l1_idx = []
    l2_idx = []
    
    for s in range(8):
        b = syms[s]
        for r_off in range(3):
            for c_off in range(2):
                shape = [b[r_off, c_off], b[r_off, c_off+1], b[r_off, c_off+2],
                         b[r_off+1, c_off], b[r_off+1, c_off+1], b[r_off+1, c_off+2]]
                rect_idx.append(shape)
                
        for r_off in range(3):
            shape = [b[r_off, 0], b[r_off, 1], b[r_off, 2], b[r_off, 3],
                     b[r_off+1, 0], b[r_off+1, 1]]
            l1_idx.append(shape)
            
            shape2 = [b[r_off, 0], b[r_off, 1], b[r_off, 2], b[r_off, 3],
                      b[r_off+1, 1], b[r_off+1, 2]]
            l2_idx.append(shape2)
            
    return np.array(rect_idx, dtype=np.int32), np.array(l1_idx, dtype=np.int32), np.array(l2_idx, dtype=np.int32)

RECT_IDX, L1_IDX, L2_IDX = get_sym_indices()

@njit(nogil=True)
def evaluate_afterstate(afterstate, tables, rect_idx, l1_idx, l2_idx):
    b = np.zeros(16, dtype=np.int32)
    for r in range(4):
        for c in range(4):
            b[r*4 + c] = afterstate[r, c]
            
    val = 0.0
    for i in range(48):
        idx = rect_idx[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        val += tables[0, f]
    for i in range(24):
        idx = l1_idx[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        val += tables[1, f]
    for i in range(24):
        idx = l2_idx[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        val += tables[2, f]
    return val

@njit(nogil=True)
def update_tables(afterstate, tables, delta, alpha, rect_idx, l1_idx, l2_idx):
    b = np.zeros(16, dtype=np.int32)
    for r in range(4):
        for c in range(4):
            b[r*4 + c] = afterstate[r, c]
            
    adj_alpha = alpha / 96.0
    
    for i in range(48):
        idx = rect_idx[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        tables[0, f] += adj_alpha * delta
    for i in range(24):
        idx = l1_idx[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        tables[1, f] += adj_alpha * delta
    for i in range(24):
        idx = l2_idx[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        tables[2, f] += adj_alpha * delta

# Board operations
@njit
def get_row_move(row):
    # row is a 1D array of 4 ints
    new_row = np.zeros(4, dtype=np.int32)
    score = 0
    idx = 0
    
    # slide and merge
    temp = np.zeros(4, dtype=np.int32)
    t_idx = 0
    for i in range(4):
        if row[i] != 0:
            temp[t_idx] = row[i]
            t_idx += 1
            
    i = 0
    while i < t_idx:
        if i < t_idx - 1 and temp[i] == temp[i+1]:
            new_row[idx] = temp[i] + 1 # increment power of 2
            score += (1 << new_row[idx])
            i += 2
        else:
            new_row[idx] = temp[i]
            i += 1
        idx += 1
        
    return new_row, score

@njit
def move_board(board, direction):
    # 0: Up, 1: Right, 2: Down, 3: Left
    new_board = np.zeros((4,4), dtype=np.int32)
    score = 0
    moved = False
    
    if direction == 3: # Left
        for r in range(4):
            new_row, s = get_row_move(board[r, :])
            new_board[r, :] = new_row
            score += s
            for c in range(4):
                if new_board[r,c] != board[r,c]:
                    moved = True
    elif direction == 1: # Right
        for r in range(4):
            row = board[r, ::-1]
            new_row, s = get_row_move(row)
            new_board[r, ::-1] = new_row
            score += s
            for c in range(4):
                if new_board[r,c] != board[r,c]:
                    moved = True
    elif direction == 0: # Up
        for c in range(4):
            row = board[:, c]
            new_row, s = get_row_move(row)
            new_board[:, c] = new_row
            score += s
            for r in range(4):
                if new_board[r,c] != board[r,c]:
                    moved = True
    elif direction == 2: # Down
        for c in range(4):
            row = board[::-1, c]
            new_row, s = get_row_move(row)
            new_board[::-1, c] = new_row
            score += s
            for r in range(4):
                if new_board[r,c] != board[r,c]:
                    moved = True
                    
    return new_board, score, moved

@njit
def spawn_tile(board):
    empty = []
    for r in range(4):
        for c in range(4):
            if board[r,c] == 0:
                empty.append((r,c))
    if len(empty) == 0:
        return board
    
    idx = np.random.randint(len(empty))
    r, c = empty[idx]
    val = 1 if np.random.random() < 0.9 else 2
    
    new_board = np.copy(board)
    new_board[r,c] = val
    return new_board

@njit(nogil=True)
def get_best_move(board, tables, rect_idx, l1_idx, l2_idx):
    best_val = -1e9
    best_move = -1
    best_afterstate = np.zeros((4,4), dtype=np.int32)
    best_score = 0
    
    for d in range(4):
        afterstate, score, moved = move_board(board, d)
        if moved:
            val = score + evaluate_afterstate(afterstate, tables, rect_idx, l1_idx, l2_idx)
            if val > best_val:
                best_val = val
                best_move = d
                best_afterstate = afterstate
                best_score = score
                
    return best_move, best_afterstate, best_score, best_val

@njit(nogil=True)
def play_game(tables, alpha, rect_idx, l1_idx, l2_idx):
    board = np.zeros((4,4), dtype=np.int32)
    board = spawn_tile(board)
    board = spawn_tile(board)
    
    total_score = 0
    moves = 0
    
    # First move
    move, afterstate, score, val = get_best_move(board, tables, rect_idx, l1_idx, l2_idx)
    if move == -1: return total_score, board
    
    while True:
        # 1. Environment responds
        next_board = spawn_tile(afterstate)
        
        # 2. Agent decides next move
        next_move, next_afterstate, next_score, next_val = get_best_move(next_board, tables, rect_idx, l1_idx, l2_idx)
        
        # 3. TD Update
        if next_move == -1:
            # Game over, next value is 0
            delta = 0 - evaluate_afterstate(afterstate, tables, rect_idx, l1_idx, l2_idx)
            update_tables(afterstate, tables, delta, alpha, rect_idx, l1_idx, l2_idx)
            break
        else:
            # TD Error
            delta = next_val - evaluate_afterstate(afterstate, tables, rect_idx, l1_idx, l2_idx)
            update_tables(afterstate, tables, delta, alpha, rect_idx, l1_idx, l2_idx)
            
            # Step forward
            afterstate = next_afterstate
            total_score += next_score
            moves += 1
            
    return total_score, next_board

def save_weights(tables, filepath):
    np.save(filepath, tables)

def load_weights(filepath):
    if os.path.exists(filepath):
        return np.load(filepath)
    return np.zeros((NUM_TABLES, TABLE_SIZE), dtype=np.float32)

def worker_batch(tables, alpha, num_games, rect_idx, l1_idx, l2_idx):
    scores = np.zeros(num_games, dtype=np.float32)
    max_tiles = np.zeros(num_games, dtype=np.int32)
    for i in range(num_games):
        score, final_board = play_game(tables, alpha, rect_idx, l1_idx, l2_idx)
        scores[i] = score
        max_tiles[i] = 1 << np.max(final_board)
    return scores, max_tiles

def train(speed, total_epochs):
    tables = load_weights("weights.npy")
    print("Compiling environment (warmup)...")
    play_game(tables, 0.0, RECT_IDX, L1_IDX, L2_IDX) # Warmup to avoid multithreading compile race conditions
    
    import multiprocessing
    import concurrent.futures
    
    max_workers = multiprocessing.cpu_count()
    if speed == 1:
        workers = 1
    elif speed == 2:
        workers = max(1, max_workers // 2)
    else:
        workers = max_workers
        
    print(f"Starting TD(0) Learning with {workers} parallel workers...")
    print("Press Ctrl+C to stop training at any time.")
    
    start_time = time.time()
    
    alpha_start = 0.0025
    alpha_end = 0.0001
    
    games_played = 0
    batch_size = 1000
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            while total_epochs is None or games_played < total_epochs:
                # Decay alpha over 10M games if infinite
                decay_target = total_epochs if total_epochs is not None else 10000000
                progress = min(1.0, games_played / decay_target)
                current_alpha = alpha_start - progress * (alpha_start - alpha_end)
                
                futures = []
                for _ in range(workers):
                    futures.append(executor.submit(worker_batch, tables, current_alpha, batch_size, RECT_IDX, L1_IDX, L2_IDX))
                    
                all_scores = []
                all_max_tiles = []
                for f in concurrent.futures.as_completed(futures):
                    scores, max_tiles = f.result()
                    all_scores.extend(scores)
                    all_max_tiles.extend(max_tiles)
                    
                games_played += workers * batch_size
                
                avg_score = np.mean(all_scores)
                max_t = np.max(all_max_tiles)
                t_ps = (workers * batch_size) / (time.time() - start_time)
                
                print(f"Games: {games_played} | Avg Score: {avg_score:.0f} | Max Tile: {max_t} | Alpha: {current_alpha:.5f} | Games/sec: {t_ps:.1f}")
                save_weights(tables, "weights.npy")
                start_time = time.time()
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
        save_weights(tables, "weights.npy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, choices=[1, 2, 3], default=3, help="Training speed (CPU usage: 1=Low, 2=Medium, 3=Max)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of games to train (default: infinite)")
    args = parser.parse_args()
    train(args.speed, args.epochs)
