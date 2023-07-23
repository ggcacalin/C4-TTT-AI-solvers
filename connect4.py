import numpy as np
import random
from itertools import cycle
import time
import math
import pickle

squares = []
for i in range(6):
    row = []
    for j in range(7):
        row.append(' ')
    squares.append(row)
dummy_row = []
for j in range(7):
    dummy_row.append('-')
squares.append(dummy_row)
minimax_calls = 0

def draw_board(squares):
    for i in range(len(squares) - 1):
        row = ''
        for j in range(len(squares[0])):
            row = row + squares[i][j] + '|'
        print('|' + row)

def reset_board(squares):
    for i in range(len(squares) - 1):
        for j in range(len(squares[0])):
            squares[i][j] = ' '

def human_move(squares, turn):
    print('Human to move as ' + turn + ' ')
    complete = False
    while not complete:
        c = input('Give column: ')
        c = int(c)
        if c not in range(len(squares[0])):
            print('Not possible, use integers between 0 and ' + str(len(squares[0]) - 1))
            continue
        if squares[0][c] == ' ':
            #Find first unfilled slot
            for slot in range(len(squares)-1):
                if squares[slot+1][c] != ' ':
                    squares[slot][c] = turn
                    complete = True
                    break
        else:
            print('Space taken, try again.')
    return [turn, c]

def evaluate_state(squares, turn, weights):
    # Executed after move was made
    player_set = {'X', 'O'}
    opponent = list(player_set - {turn})[0]
    value = 0
    # Weights vector: gains 3/2, losses 3/2
    w_p_3 = weights[0]
    w_p_2 = weights[1]
    w_o_3 = weights[2]
    w_o_2 = weights[3]

    # Checking rows
    for r in reversed(range(len(squares) - 1)):
        for c in range(int(len(squares[0]) + 1 / 2)):
            if squares[r][c:(c+4)].count(turn) == 3 and squares[r][c:(c+4)].count(' ') == 1:
                value += w_p_3
            if squares[r][c:(c + 4)].count(turn) == 2 and squares[r][c:(c + 4)].count(' ') == 2:
                value += w_p_2
            if squares[r][c:(c + 4)].count(opponent) == 3 and\
                    squares[r][c:(c + 4)].count(' ') == 1:
                value += w_o_3
            if squares[r][c:(c + 4)].count(opponent) == 2 and\
                    squares[r][c:(c + 4)].count(' ') == 2:
                value += w_o_2
    # Checking columns
    for c in range(len(squares[0])):
        for r in reversed(range(3, len(squares) - 1)):
            relevant_range = []
            for i in range(4):
                relevant_range.append(squares[r-i][c])
            if relevant_range.count(turn) == 3 and relevant_range.count(' ') == 1:
                value += w_p_3
            if relevant_range.count(turn) == 2 and relevant_range.count(' ') == 2:
                value += w_p_2
            if relevant_range.count(opponent) == 3 and relevant_range.count(' ') == 1:
                value += w_o_3
            if relevant_range.count(opponent) == 2 and relevant_range.count(' ') == 2:
                value += w_o_2
    # Checking diagonals
    for r in reversed(range(3, len(squares) - 1)):
        #Secondary diagonals
        for c in range(4):
            relevant_range = []
            for i in range(4):
                relevant_range.append(squares[r-i][c+i])
            if relevant_range.count(turn) == 3 and relevant_range.count(' ') == 1:
                value += w_p_3
            if relevant_range.count(turn) == 2 and relevant_range.count(' ') == 2:
                value += w_p_2
            if relevant_range.count(opponent) == 3 and relevant_range.count(' ') == 1:
                value += w_o_3
            if relevant_range.count(opponent) == 2 and relevant_range.count(' ') == 2:
                value += w_o_2
        #Primary diagonals
        for c in range(3, 7):
            relevant_range = []
            for i in range(4):
                relevant_range.append(squares[r-i][c-i])
            if relevant_range.count(turn) == 3 and relevant_range.count(' ') == 1:
                value += w_p_3
            if relevant_range.count(turn) == 2 and relevant_range.count(' ') == 2:
                value += w_p_2
            if relevant_range.count(opponent) == 3 and relevant_range.count(' ') == 1:
                value += w_o_3
            if relevant_range.count(opponent) == 2 and relevant_range.count(' ') == 2:
                value += w_o_2

    return value

def row_check_win(squares, turn):
    for r in reversed(range(len(squares) - 1)):
        for c in range(int(len(squares[0]) + 1 / 2)):
            if squares[r][c:(c+4)].count(turn) == 3 and squares[r][c:(c+4)].count(' ') == 1:
                if squares[r+1][c + squares[r][c:(c+4)].index(' ')] != ' ':
                    j = squares[r][c:(c+4)].index(' ')
                    squares[r][c + j] = turn
                    move = [r, c + j]
                    return move

def col_check_win(squares, turn):
    for c in range(len(squares[0])):
        for r in reversed(range(3, len(squares) - 1)):
            relevant_range = []
            for i in range(4):
                relevant_range.append(squares[r-i][c])
            if relevant_range.count(turn) == 3 and relevant_range.count(' ') == 1:
                squares[r-3][c] = turn
                move = [r-3, c]
                return move

def diag_check_win(squares, turn):
    for r in reversed(range(3, len(squares) - 1)):
        #Secondary diagonals
        for c in range(4):
            relevant_range = []
            for i in range(4):
                relevant_range.append(squares[r-i][c+i])
            if relevant_range.count(turn) == 3 and relevant_range.count(' ') == 1:
                if squares[r-relevant_range.index(' ')+1][c+relevant_range.index(' ')] != ' ':
                    j = relevant_range.index(' ')
                    squares[r - j][c + j] = turn
                    move = [r - j, c + j]
                    return move
        #Primary diagonals
        for c in range(3, 7):
            relevant_range = []
            for i in range(4):
                relevant_range.append(squares[r-i][c-i])
            if relevant_range.count(turn) == 3 and relevant_range.count(' ') == 1:
                if squares[r-relevant_range.index(' ')+1][c-relevant_range.index(' ')] != ' ':
                    squares[r - relevant_range.index(' ')][c - relevant_range.index(' ')] = turn
                    move = [r - relevant_range.index(' '), c - relevant_range.index(' ')]
                    return move

def baseline_move(squares, turn):
    move = []
    player_set = {'X', 'O'}
    # Do winning move if it exists
    move = row_check_win(squares, turn)
    if not move:
        move = col_check_win(squares, turn)
    if not move:
        move = diag_check_win(squares, turn)
    # Counter if possible
    if not move:
        move = row_check_win(squares, list(player_set - {turn})[0])
        if move:
            squares[move[0]][move[1]] = turn
    if not move:
        move = col_check_win(squares, list(player_set - {turn})[0])
        if move:
            squares[move[0]][move[1]] = turn
    if not move:
        move = diag_check_win(squares, list(player_set - {turn})[0])
        if move:
            squares[move[0]][move[1]] = turn
    # Do random move if nothing smarter:
    if not move:
        possible_moves = []
        for c in range(len(squares[0])):
            if squares[0][c] == ' ':
                possible_moves.append(c)
        m = random.choice(possible_moves)
        for r in range(len(squares)-1):
            if squares[r+1][m] != ' ':
                squares[r][m] = turn
                break
        move = [r, m]
    return [turn, move]

def check_win(squares):
    #Rows
    for i in range(len(squares)-1):
        for j in range(len(squares[0]) - 3):
            if squares[i][j] == squares[i][j+1] == squares[i][j+2] == squares[i][j+3] and\
                squares[i][j] != ' ':
                return 1, squares[i][j]

    #Columns
    for j in range(len(squares[0])):
        for i in range(len(squares) - 4):
            if squares[i][j] == squares[i+1][j] == squares[i+2][j] == squares[i+3][j] and\
                    squares[i][j] != ' ':
                return 1, squares[i][j]

    #Right-leaning diagonals
    for i in reversed(range(3, len(squares) - 1)):
        for j in range(0, int((len(squares[0]) + 1 )/ 2)):
            if squares[i][j] == squares[i-1][j+1] == squares[i-2][j+2] == squares[i-3][j+3] and\
                squares[i][j] != ' ':
                return 1, squares[i][j]

    #Left-leaning diagonals
    for i in reversed(range(3, len(squares) - 1)):
        for j in reversed(range(int(len(squares[0]) + 1 / 2) - 1, len(squares[0]))):
            if squares[i][j] == squares[i-1][j-1] == squares[i-2][j-2] == squares[i-3][j-3] and\
                squares[i][j] != ' ':
                return 1, squares[i][j]

    #Tie
    if not any(' ' in row for row in squares[:(len(squares) - 1)]):
        return 1, 'T'
    return 0, None

def bad_max_value(squares, turn, alpha, beta):
    # Maximizing own score
    global minimax_calls
    minimax_calls+= 1
    print(minimax_calls)
    value = -math.inf
    player_set = {'X', 'O'}
    r = None
    c = None
    # Check ending conditions
    result = check_win(squares)
    if result[1] == turn:
        return [10, 0, 0]
    if result[1] == list(player_set - {turn})[0]:
        return [-10, 0, 0]
    if result[1] == 'T':
        return [0, 0, 0]
    # Build tree
    for i in range(len(squares) - 1):
        for j in range(len(squares[0])):
            if squares[i][j] == ' ' and squares[i + 1][j] != ' ':
                squares[i][j] = turn
                [m, min_r, min_c] = bad_min_value(squares, list(player_set - {turn})[0], alpha, beta)
                # Update value and undo analysis board change
                if m > value:
                    value = m
                    r = i
                    c = j
                squares[i][j] = ' '
                if value >= beta:
                    return [value, r, c]
                alpha = max(alpha, value)
    return [value, r, c]

def bad_min_value(squares, turn, alpha, beta):
    # Maximizing own score
    global minimax_calls
    minimax_calls += 1
    print(minimax_calls)
    value = math.inf
    player_set = {'X', 'O'}
    r = None
    c = None
    # Check ending conditions
    result = check_win(squares)
    if result[1] == turn:
        return [-10 ** 9, 0, 0]
    if result[1] == list(player_set - {turn})[0]:
        return [10 ** 9, 0, 0]
    if result[1] == 'T':
        return [0, 0, 0]
    # Build tree
    for i in range(len(squares) - 1):
        for j in range(len(squares[0])):
            if squares[i][j] == ' ' and squares[i + 1][j] != ' ':
                squares[i][j] = turn
                [m, max_r, max_c] = bad_max_value(squares, list(player_set - {turn})[0], alpha, beta)
                # Updates
                if m < value:
                    value = m
                    r = i
                    c = j
                squares[i][j] = ' '
                # Pruning and updating beta
                if value <= alpha:
                    return [value, r, c]
                beta = min(value, beta)
    return [value, r, c]

def max_value(squares, turn, depth, weights, alpha, beta):
    #Maximizing own score
    value = -math.inf
    player_set = {'X', 'O'}
    r = None
    c = None
    # Check ending conditions
    result = check_win(squares)
    if result[1] == turn:
        return [10**9, 0, 0]
    if result[1] == list(player_set - {turn})[0]:
        return [-10**9, 0, 0]
    if result[1] == 'T':
        return [0, 0, 0]
    if depth == 0:
        return [evaluate_state(squares, turn, weights), 0, 0]
    # Build tree
    for i in range(len(squares) - 1):
        for j in range(len(squares[0])):
            if squares[i][j] == ' ' and squares[i+1][j] != ' ':
                squares[i][j] = turn
                [m, min_r, min_c] = min_value(squares, list(player_set - {turn})[0], depth-1, weights, alpha, beta)
                #Update value and undo analysis board change
                if m > value:
                    value = m
                    r = i
                    c = j
                squares[i][j] = ' '
                # Pruning and updating alpha
                if value >= beta:
                    return [value, r, c]
                alpha = max(alpha, value)
    return [value, r, c]

def min_value(squares, turn, depth, weights, alpha, beta):
    # Maximizing own score
    value = math.inf
    player_set = {'X', 'O'}
    r = None
    c = None
    # Check ending conditions
    result = check_win(squares)
    if result[1] == turn:
        return [-10 ** 9, 0, 0]
    if result[1] == list(player_set - {turn})[0]:
        return [10 ** 9, 0, 0]
    if result[1] == 'T':
        return [0, 0, 0]
    if depth == 0:
        return [evaluate_state(squares, turn, weights), 0, 0]
    # Build tree
    for i in range(len(squares) - 1):
        for j in range(len(squares[0])):
            if squares[i][j] == ' ' and squares[i + 1][j] != ' ':
                squares[i][j] = turn
                [m, max_r, max_c] = max_value(squares, list(player_set - {turn})[0], depth-1, weights, alpha, beta)
                # Updates
                if m < value:
                    value = m
                    r = i
                    c = j
                squares[i][j] = ' '
                # Pruning and updating beta
                if value <= alpha:
                    return [value, r, c]
                beta = min(value, beta)
    return [value, r, c]

def hash_qargs(squares, turn, move):
    row_list = []
    for row in squares:
        row_list.append(tuple(row))
    row_list = tuple(row_list)
    turn_move_tuple = tuple([turn, tuple(move)])
    return (row_list, turn_move_tuple)

def get_Q(squares, turn, qvals, move):
    key = hash_qargs(squares, turn, move)
    if key not in qvals:
        qvals[key] = 0.0
    return qvals[key]

def QTrain(squares, turn, opponent, alpha, gamma, epsilon, iterations, depth, weights):
    qvals = {}
    # Load old epoch if it exists
    if epsilon < 1 or opponent != 'baseline':
        if turn == 'X':
            with open('Q_C4_X.pickle', 'rb') as jar:
                qvals = pickle.load(jar)
        else:
            with open('Q_C4_O.pickle', 'rb') as jar:
                qvals = pickle.load(jar)
    player = turn
    for k in range(iterations):
        players = cycle(['X', 'O'])
        status = False
        winner = None
        turn = next(players)
        moves = []
        # Run game against baseline
        while (status == False):
            # Gather available moves
            if turn == player:
                available_moves = []
                for i in range(6):
                    for j in range(7):
                        if squares[i][j] == ' ' and squares[i+1][j] != ' ':
                            available_moves.append([i, j])
                # Act upon available moves
                if (random.uniform(0,1) < epsilon):
                    move = random.choice(available_moves)
                    squares[move[0]][move[1]] = turn
                else:
                    q_list = []
                    for move in available_moves:
                        q_list.append(get_Q(squares, turn, qvals, move))
                    max_Q = max(q_list)
                    # Choose random move if multiple have the same value, best otherwise
                    if q_list.count(max_Q) > 1:
                        options = [i for i in range(len(available_moves)) if q_list[i] == max_Q]
                        index = random.choice(options)
                    else:
                        index = q_list.index(max_Q)
                    move = available_moves[index]
                    squares[move[0]][move[1]] = turn
            else:
                if opponent == 'baseline':
                    move = baseline_move(squares, turn)[1]
                if opponent == 'minimax':
                    [value, r, c] = max_value(squares, turn, depth, weights, -math.inf, math.inf)
                    squares[r][c] = turn
                    move = [r, c]
            moves.append([turn, move])
            status, winner = check_win(squares)
            turn = next(players)
        # Equate final Q value to reward instead of updating it
        if winner == player:
            reward = 100
        elif winner == 'T':
            reward = 50
        else:
            reward = -100
        final_move = True
        while(len(moves) > 0):
            # Remove opponent move if it exists
            if moves[-1][0] != player:
                squares[moves[-1][1][0]][moves[-1][1][1]] = ' '
                moves = moves[:-1]
                continue
            # Update q value of player's moves
            if final_move:
                # If move led to the end of the game, use reward
                squares[moves[-1][1][0]][moves[-1][1][1]] = ' '
                qvals[hash_qargs(squares, player, moves[-1][1])] =(1-alpha) *\
                                                                  get_Q(squares, player, qvals, moves[-1][1]) +\
                                             alpha * (reward + gamma * reward)
                moves = moves[:-1]
                final_move = False
            else:
                # Remove and remember last player move
                last_move = moves[-1]
                moves = moves[:-1]
                s_prime = squares.copy()
                squares[last_move[1][0]][last_move[1][1]] = ' '
                # Find maximum q-value over actions in the post-move state
                potential_values = []
                for i in range(6):
                    for j in range(7):
                        if squares[i][j] == ' ' and squares[i+1][j] != ' ':
                            potential_values.append(get_Q(s_prime, player, qvals, [i, j]))
                # Use maximum q-value and last move to update q-value of pre-move state
                qvals[hash_qargs(squares, player, last_move[1])] = (1-alpha) *\
                                                                   get_Q(squares, player, qvals, last_move[1]) +\
                                              alpha * (reward + gamma * max(potential_values))
    # Store the q-value table
    print('Length of ' + player + ' player table: ' + str(len(qvals)))
    filename = 'Q_C4_' + player + '.pickle'
    with open(filename, 'wb') as jar:
        pickle.dump(qvals, jar, protocol=pickle.HIGHEST_PROTOCOL)

# Train Q player for X, O
def train_both():
    depth = 6
    weights = [10, 5, -8, -4]
    for epoch in range(100):
        print('Epoch: ' + str(epoch))
        QTrain(squares, 'X', 'baseline', 0.9*0.9**epoch, 0.9, 0.8**epoch, 5000, depth, weights)
        QTrain(squares, 'O', 'baseline', 0.9*0.9**epoch, 0.9, 0.8**epoch, 5000, depth, weights)

def QPlayer_move(squares, turn):
    print('QPlayer to move as ' + turn)
    # Find available moves
    available_moves = []
    for i in range(6):
        for j in range(7):
            if squares[i][j] == ' ' and squares[i+1][j] != ' ':
                available_moves.append([i, j])
    # Find and execute highest q-value move
    potential_values = []
    for move in available_moves:
        if turn == 'X':
            potential_values.append(get_Q(squares, turn, qvalues_X, move))
        else:
            potential_values.append(get_Q(squares, turn, qvalues_O, move))
    chosen_move = available_moves[potential_values.index(max(potential_values))]
    squares[chosen_move[0]][chosen_move[1]] = turn
    return [turn, chosen_move]

def play(p_1, p_2, squares, depth, weights):
    players = cycle(['X', 'O'])
    winner = None
    turn = next(players)
    draw_board(squares)
    time_history_1 = []
    time_history_2 = []
    #True status means game has concluded
    status = False
    if (p_1 == 'B' and p_2 == 'M'):
        while (status == False):
            if turn == 'X':
                print('Baseline to move as ' + turn)
                baseline_move(squares, turn)
            else:
                print('Minimax to move as ' + turn)
                tic = time.time()
                [value, r, c] = max_value(squares, turn, depth, weights, -math.inf, math.inf)
                toc = time.time()
                time_history_1.append(round((toc - tic) * 1000.0))
                squares[r][c] = turn
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'M' and p_2 == 'B'):
        while (status == False):
            if turn == 'X':
                print('Minimax to move as ' + turn)
                tic = time.time()
                [value, r, c] = max_value(squares, turn, depth, weights, -math.inf, math.inf)
                toc = time.time()
                time_history_1.append(round((toc - tic) * 1000.0))
                squares[r][c] = turn
            else:
                print('Baseline to move as ' + turn)
                baseline_move(squares, turn)
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'B' and p_2 == 'Q'):
        while (status == False):
            if turn == 'X':
                print('Baseline to move as ' + turn)
                baseline_move(squares, turn)
            else:
                print('QPlayer to move as ' + turn)
                tic = time.time()
                QPlayer_move(squares, turn)
                time_history_1.append(round((time.time() - tic) * 1000.0))
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'Q' and p_2 == 'B'):
        while (status == False):
            if turn == 'X':
                print('QPlayer to move as ' + turn)
                tic = time.time()
                current_move = QPlayer_move(squares, turn)
                time_history_1.append(round((time.time() - tic) * 1000.0))
            else:
                print('Baseline to move as ' + turn)
                current_move = baseline_move(squares, turn)
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'Q' and p_2 == 'M'):
        while (status == False):
            if turn == 'X':
                QPlayer_move(squares, turn)
            else:
                [value, r, c] = max_value(squares, turn, depth, weights, -math.inf, math.inf)
                squares[r][c] = turn
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'M' and p_2 == 'Q'):
        while (status == False):
            if turn == 'X':
                [value, r, c] = max_value(squares, turn, depth, weights, -math.inf, math.inf)
                squares[r][c] = turn
            else:
                QPlayer_move(squares, turn)
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'TM' and p_2 == 'B'):
        while (status == False):
            if turn == 'X':
                [value, r, c] = bad_max_value(squares, turn, -math.inf, math.inf)
                squares[r][c] = turn
            else:
                baseline_move(squares, turn)
                global minimax_calls
                minimax_calls = 0
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
    return winner, time_history_1

#train_both()
with open('Q_C4_X.pickle', 'rb') as jar:
    qvalues_X = pickle.load(jar)
with open('Q_C4_O.pickle', 'rb') as jar:
    qvalues_O = pickle.load(jar)
depth = 6
weights = [10, 5, -8, -4]
X_win_history = {'X': 0, 'O': 0, 'T': 0}
for i in range(1):
    winner, times_X = play('M', 'Q', squares, depth, weights)
    if winner is not None:
        X_win_history[winner] += 1
O_win_history = {'X': 0, 'O': 0, 'T': 0}
for i in range(1):
    winner, times_O = play('Q', 'M', squares, depth, weights)
    if winner is not None:
        O_win_history[winner] += 1
print('X player outcomes:')
print(X_win_history)
#print(np.mean(times_X))
print('O player outcomes:')
print(O_win_history)
#print(np.mean(times_O))
