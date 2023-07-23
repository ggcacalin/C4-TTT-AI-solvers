import numpy as np
import random
import math
import time
import pickle
from itertools import cycle

# Set up underlying board
squares = []
for i in range(3):
    row = []
    for j in range(3):
        row.append(' ')
    squares.append(row)

# Prints the current board in console
def draw_board(squares):
    for i in range(3):
        print('|' + squares[i][0] + '|' + squares[i][1] + '|' + squares[i][2] + '|')

def reset_board(squares):
    for i in range(len(squares)):
        for j in range(len(squares[0])):
            squares[i][j] = ' '

def human_move(squares, turn):
    print('Human to move as ' + turn + ' ')
    complete = False
    while not complete:
        i, j = input('Give row and column: ').split()
        i = int(i); j = int(j)
        if i not in range(len(squares)) or j not in range(len(squares[0])):
            print('Not possible, use integers between 0 and 8.')
        if squares[i][j] == ' ':
            squares[i][j] = turn
            complete = True
        else:
            print('Space taken, try again.')
    return [turn, [i, j]]

def baseline_move(squares, turn):
    #print('Baseline to move as ' + turn + ' ')
    move = []
    #Do winning move if it exists
    for r in range(len(squares)):
        for c in range(len(squares[0])):
            if squares[r][c] == ' ':
                if squares[r][(c+1)%3] == turn and squares[r][(c+2)%3] == turn:
                    squares[r][c] = turn
                    move = [r, c]
                    return [turn, move]
                if squares[(r+1)%3][c] == turn and squares[(r+2)%3][c] == turn:
                    squares[r][c] = turn
                    move = [r, c]
                    return [turn, move]
                if r == c and squares[(r+1)%3][(c+1)%3] == turn and squares[(r+2)%3][(c+2)%3] == turn:
                    squares[r][c] = turn
                    move = [r, c]
                    return [turn, move]
                if r == len(squares) - c - 1 and squares[(r-1)%3][(c-1)%3] == turn and\
                        squares[(r-2)%3][(c-2)%3] == turn:
                    squares[r][c] = turn
                    move = [r, c]
                    return [turn, move]
    #Counter if possible
    if not move:
        for r in range(len(squares)):
            for c in range(len(squares[0])):
                if squares[r][c] == ' ':
                    if squares[r][(c + 1) % 3] not in [' ', turn] and\
                            squares[r][(c + 2) % 3] not in [' ', turn]:
                        squares[r][c] = turn
                        move = [r, c]
                        return [turn, move]
                    if squares[(r + 1) % 3][c] not in [' ', turn] and\
                            squares[(r + 2) % 3][c] not in [' ', turn]:
                        squares[r][c] = turn
                        move = [r, c]
                        return [turn, move]
                    if r == c and squares[(r + 1) % 3][(c + 1) % 3] not in [' ', turn] and\
                            squares[(r + 2) % 3][(c + 2) % 3] not in [' ', turn]:
                        squares[r][c] = turn
                        move = [r, c]
                        return [turn, move]
                    if r == len(squares) - c - 1 and squares[(r - 1) % 3][(c - 1) % 3] not in [' ', turn] and\
                            squares[(r - 2) % 3][(c - 2) % 3] not in [' ', turn]:
                        squares[r][c] = turn
                        move = [r, c]
                        return [turn, move]
    #Do random move if nothing smarter:
    if not move:
        possible_moves = []
        for r in range(len(squares)):
            for c in range(len(squares[0])):
                if squares[r][c] == ' ':
                    possible_moves.append([r, c])
        m = random.choice(possible_moves)
        squares[m[0]][m[1]] = turn
        move = [m[0], m[1]]
    return [turn, move]

def check_win(squares):
    #Rows
    for i in range(len(squares)):
        if squares[i][0] == squares[i][1] == squares[i][2] and\
            squares[i][0] != ' ':
            return 1, squares[i][0]

    #Columns
    for i in range(len(squares[0])):
        if squares[0][i] == squares[1][i] == squares[2][i] and\
                squares[0][i] != ' ':
            return 1, squares[0][i]
    #Diagonals
    if (len(np.unique(np.diag(squares))) == 1 or\
        len(np.unique(np.diag(np.fliplr(squares)))) == 1) and\
            squares[1][1] != ' ':
        return 1, squares[1][1]

    #Tie
    if not any(' ' in row for row in squares):
        return 1, 'T'
    return 0, None

def max_value(squares, turn, alpha, beta):
    #Maximizing own score
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

    for i in range(3):
        for j in range(3):
            if squares[i][j] == ' ':

                squares[i][j] = turn
                [m, min_r, min_c] = min_value(squares, list(player_set - {turn})[0], alpha, beta)
                #Update value and undo analysis board change
                if m > value:
                    value = m
                    r = i
                    c = j
                squares[i][j] = ' '
                # Maximiser: pruning and updating alpha
                if value >= beta:
                    return [value, r, c]
                alpha = max(alpha, value)
    return [value, r, c]

def min_value(squares, turn, alpha, beta):
    #Minimizing adverasry score
    value = math.inf
    player_set = {'X', 'O'}
    r = None
    c = None
    #Check ending conditions
    result = check_win(squares)
    if result[1] == turn:
        return [-10, 0, 0]
    if result[1] == list(player_set - {turn})[0]:
        return [10, 0, 0]
    if result[1] == 'T':
        return [0, 0, 0]
    # Build tree
    for i in range(3):
        for j in range(3):
            if squares[i][j] == ' ':
                squares[i][j] = turn
                [m, max_r, max_c] = max_value(squares, list(player_set - {turn})[0], alpha, beta)
                # Updates
                if m < value:
                    value = m
                    r = i
                    c = j
                squares[i][j] = ' '
                # Minimiser: pruning and updating beta
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

def QTrain(squares, turn, opponent, alpha, gamma, epsilon, iterations):
    qvals = {}
    # Load old epoch if it exists
    if epsilon < 0.8 or opponent != 'baseline':
        if turn == 'X':
            with open('Q_TTT_X.pickle', 'rb') as jar:
                qvals = pickle.load(jar)
        else:
            with open('Q_TTT_O.pickle', 'rb') as jar:
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
                for i in range(3):
                    for j in range(3):
                        if squares[i][j] == ' ':
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
                    [value, r, c] = max_value(squares, turn, -math.inf, math.inf)
                    squares[r][c] = turn
                    move = [r, c]
            moves.append([turn, move])
            status, winner = check_win(squares)
            turn = next(players)
        # Equate final Q value to reward instead of updating it
        if winner == player:
            reward = 1
        elif winner == 'T':
            reward = 0.5
        else:
            reward = -1
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
                                             alpha * (reward + gamma*reward)
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
                for i in range(3):
                    for j in range(3):
                        if squares[i][j] == ' ':
                            potential_values.append(get_Q(s_prime, player, qvals, [i, j]))
                # Use maximum q-value and last move to update q-value of pre-move state
                qvals[hash_qargs(squares, player, last_move[1])] = (1-alpha) *\
                                                                   get_Q(squares, player, qvals, last_move[1]) +\
                                              alpha *(reward + gamma * max(potential_values))
    # Store the q-value table
    print('Length of ' + player + ' player table: ' + str(len(qvals)))
    filename = 'Q_TTT_' + player + '.pickle'
    with open(filename, 'wb') as jar:
        pickle.dump(qvals, jar, protocol=pickle.HIGHEST_PROTOCOL)

def train_both():
# Train Q player for X, O
    for epoch in range(50):
        print('Epoch: ' + str(epoch))
        QTrain(squares, 'X', 'baseline', 0.8*0.95**epoch, 0.9, 0.8*0.95**epoch, 400)
        QTrain(squares, 'X', 'minimax', 0.8 * 0.95 ** epoch, 0.9, 0.8 * 0.95 ** epoch, 100)
        QTrain(squares, 'O', 'baseline', 0.8*0.95**epoch, 0.9, 0.8*0.95**epoch, 400)
        QTrain(squares, 'O', 'minimax', 0.8 * 0.95 ** epoch, 0.9, 0.8 * 0.95 ** epoch, 100)

def QPlayer_move(squares, turn):
    #print('QPlayer to move as ' + turn)
    # Find available moves
    available_moves = []
    for i in range(3):
        for j in range(3):
            if squares[i][j] == ' ':
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

def play(p_1, p_2, squares):
    players = cycle(['X', 'O'])
    winner = None
    turn = next(players)
    draw_board(squares)
    #True status means game has concluded
    status = False
    time_history_1 = []
    time_history_2 = []
    if (p_1 == 'B' and p_2 == 'M'):
        while (status == False):
            if turn == 'X':
                print('Baseline to move as ' + turn)
                baseline_move(squares, turn)
            else:
                print('Minimax to move as ' + turn)
                tic = time.time()
                [value, r, c] = max_value(squares, turn, -math.inf, math.inf)
                toc = time.time()
                time_history_1.append(round((toc-tic)*1000.0))
                squares[r][c] = turn
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'M' and p_2 == 'B'):
        while(status == False):
            if turn == 'X':
                print('Minimax to move as ' + turn)
                tic = time.time()
                [value, r, c] = max_value(squares, turn, -math.inf, math.inf)
                toc = time.time()
                time_history_1.append(round((toc-tic)*1000.0))
                squares[r][c] = turn
            else:
                print('Baseline to move as ' + turn)
                baseline_move(squares, turn)
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
                QPlayer_move(squares, turn)
                toc = time.time()
                time_history_1.append(round((toc - tic) * 1000.0))
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
                current_move = baseline_move(squares, turn)
            else:
                print('QPlayer to move as ' + turn)
                tic = time.time()
                QPlayer_move(squares, turn)
                toc = time.time()
                time_history_1.append(round((toc - tic) * 1000.0))
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'M' and p_2 == 'Q'):
        while(status == False):
            if turn == 'X':
                print('Minimax to move as ' + turn)
                [value, r, c] = max_value(squares, turn, -math.inf, math.inf)
                squares[r][c] = turn
            else:
                print('QPlayer to move as ' + turn)
                QPlayer_move(squares, turn)
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    if (p_1 == 'Q' and p_2 == 'M'):
        while(status == False):
            if turn == 'X':
                print('QPlayer to move as ' + turn)
                QPlayer_move(squares, turn)
            else:
                print('Minimax to move as ' + turn)
                [value, r, c] = max_value(squares, turn, -math.inf, math.inf)
                squares[r][c] = turn
            draw_board(squares)
            status, winner = check_win(squares)
            turn = next(players)
        reset_board(squares)
        print('---------------------------------------------------')
    return winner, time_history_1

train_both()
with open('Q_TTT_X.pickle', 'rb') as jar:
    qvalues_X = pickle.load(jar)
with open('Q_TTT_O.pickle', 'rb') as jar:
    qvalues_O = pickle.load(jar)

X_win_history = {'X': 0, 'O': 0, 'T': 0}
for i in range(10):
    winner, times_X = play('Q', 'M', squares)
    if winner is not None:
        X_win_history[winner] += 1
O_win_history = {'X': 0, 'O': 0, 'T': 0}
for i in range(10):
    winner, times_O = play('M', 'Q', squares)
    if winner is not None:
        O_win_history[winner] += 1

print('X player outcomes:')
print(X_win_history)
#print(np.mean(times_X))
print('O player outcomes:')
print(O_win_history)
#print(np.mean(times_O))


