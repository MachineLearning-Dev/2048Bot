import time
import os
import numpy as np
from playwright.sync_api import sync_playwright

# Pre-compute left moves for rows to speed up search
MOVE_CACHE = {}

def get_row_move_left(row):
    if row in MOVE_CACHE:
        return MOVE_CACHE[row]
    
    new_row = [x for x in row if x != 0]
    score = 0
    i = 0
    while i < len(new_row) - 1:
        if new_row[i] == new_row[i+1]:
            new_row[i] *= 2
            score += new_row[i]
            del new_row[i+1]
        i += 1
    new_row = tuple(new_row + [0] * (4 - len(new_row)))
    
    MOVE_CACHE[row] = (new_row, score)
    return new_row, score

def rot_right(board):
    return tuple(tuple(board[3-j][i] for j in range(4)) for i in range(4))

def rot_left(board):
    return tuple(tuple(board[j][3-i] for j in range(4)) for i in range(4))

def rot_180(board):
    return tuple(tuple(board[3-i][3-j] for j in range(4)) for i in range(4))

def move_left(board):
    new_board = []
    total_score = 0
    for row in board:
        r, s = get_row_move_left(row)
        new_board.append(r)
        total_score += s
    new_board = tuple(new_board)
    return new_board, total_score, new_board != board

def move_right(board):
    board = rot_180(board)
    new_board, score, moved = move_left(board)
    return rot_180(new_board), score, moved

def move_up(board):
    board = rot_left(board)
    new_board, score, moved = move_left(board)
    return rot_right(new_board), score, moved

def move_down(board):
    board = rot_right(board)
    new_board, score, moved = move_left(board)
    return rot_left(new_board), score, moved

MOVES = [move_up, move_right, move_down, move_left]
MOVE_KEYS = ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"]

LOG2_MAP = {0: 0}
for i in range(1, 18):
    LOG2_MAP[2**i] = i

EXPECTIMAX_CACHE = {}

if os.path.exists("weights.npy"):
    ML_WEIGHTS = np.load("weights.npy")
else:
    ML_WEIGHTS = np.zeros((3, 16777216))

def pack_6(a, b, c, d, e, f):
    return (a << 20) | (b << 16) | (c << 12) | (d << 8) | (e << 4) | f

def get_sym_indices():
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

def evaluate(board):
    b = [LOG2_MAP[board[r][c]] for r in range(4) for c in range(4)]
    val = 0.0
    
    for i in range(48):
        idx = RECT_IDX[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        val += ML_WEIGHTS[0, f]
    for i in range(24):
        idx = L1_IDX[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        val += ML_WEIGHTS[1, f]
    for i in range(24):
        idx = L2_IDX[i]
        f = pack_6(b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]], b[idx[4]], b[idx[5]])
        val += ML_WEIGHTS[2, f]
            
    empty = sum(1 for cell in b if cell == 0)
    return float(val) + (empty * 10.0)

def expectimax(board, depth, is_player):
    if depth == 0:
        return evaluate(board)
        
    state = (board, depth, is_player)
    if state in EXPECTIMAX_CACHE:
        return EXPECTIMAX_CACHE[state]
    
    if is_player:
        max_score = -1
        for m in MOVES:
            new_board, _, moved = m(board)
            if moved:
                score = expectimax(new_board, depth - 1, False)
                if score > max_score:
                    max_score = score
        result = max_score if max_score != -1 else evaluate(board)
    else:
        empty_cells = []
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    empty_cells.append((r, c))
        
        if not empty_cells:
            result = evaluate(board)
        else:
            expected_score = 0
            cells_to_evaluate = empty_cells
            if len(empty_cells) > 8 and depth > 1:
                 cells_to_evaluate = empty_cells[:8]
            prob_2 = 0.9 / len(cells_to_evaluate)
            prob_4 = 0.1 / len(cells_to_evaluate)

            for r, c in cells_to_evaluate:
                new_board = list(list(row) for row in board)
                new_board[r][c] = 2
                expected_score += prob_2 * expectimax(tuple(tuple(row) for row in new_board), depth - 1, True)
                new_board[r][c] = 4
                expected_score += prob_4 * expectimax(tuple(tuple(row) for row in new_board), depth - 1, True)
                
            result = expected_score

    EXPECTIMAX_CACHE[state] = result
    return result

def get_best_move(board, depth=3):
    global EXPECTIMAX_CACHE
    EXPECTIMAX_CACHE.clear()
    
    best_move = -1
    max_score = -1
    
    for i, m in enumerate(MOVES):
        new_board, _, moved = m(board)
        if moved:
            score = expectimax(new_board, depth - 1, False)
            if score > max_score:
                max_score = score
                best_move = i
                
    return best_move

def parse_board(page):
    board = [[0]*4 for _ in range(4)]
    classes_list = page.evaluate('() => Array.from(document.querySelectorAll(".tile")).map(el => el.className)')
    
    for classes_str in classes_list:
        classes = classes_str.split()
        val = 0
        pos = (0, 0)
        for c in classes:
            if c.startswith("tile-") and not c.startswith("tile-position-") and c != "tile-super":
                try:
                    val = int(c.split("-")[1])
                except ValueError:
                    pass
            if c.startswith("tile-position-"):
                parts = c.split("-")
                col = int(parts[2]) - 1
                row = int(parts[3]) - 1
                pos = (row, col)
        
        if val > board[pos[0]][pos[1]]:
            board[pos[0]][pos[1]] = val
            
    return tuple(tuple(row) for row in board)

def setup_browser(playwright_context):
    browser = playwright_context.chromium.launch(headless=False, args=['--window-size=600,800'])
    context = browser.new_context(viewport={'width': 600, 'height': 800})
    page = context.new_page()
    page.goto("https://2048game.com/")
    
    try:
        page.locator(".cookie-notice-dismiss-button").click(timeout=1000)
    except:
        pass
        
    page.wait_for_selector(".tile", state="attached", timeout=10000)
    return browser, context, page

def play_game_loop(page):
    game_over = False
    moves_made = 0
    start_time = time.time()
    
    max_tile = 0
    while not game_over:
        board = parse_board(page)
        max_tile = max(max(row) for row in board)
        
        if page.locator(".game-message.game-over").is_visible():
            break
        if page.locator(".game-message.game-won").is_visible():
            try:
                page.locator(".keep-playing-button").click(timeout=1000)
            except:
                pass
        
        empty = sum(1 for row in board for cell in row if cell == 0)
        
        depth = 3
        if empty <= 6:
            depth = 4
        if empty <= 3:
            depth = 5
            
        best_move = get_best_move(board, depth=depth)
        
        if best_move == -1:
            break
            
        key = MOVE_KEYS[best_move]
        page.keyboard.press(key)
        
        time.sleep(0.05)
        moves_made += 1
        
        if moves_made % 50 == 0:
            print(f"Moves made: {moves_made}. Max tile so far: {max_tile}")

    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")
    print(f"Total moves: {moves_made}")
    
    score = 0
    try:
        score_text = page.locator(".score-container").inner_text().split("\n")[0]
        score = int(score_text)
        print(f"Final Score: {score}")
    except:
        pass
        
    return score, max_tile
