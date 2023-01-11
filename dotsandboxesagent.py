#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxesagent.py

Template for the Machine Learning Project course at KU Leuven (2017-2018)
of Hendrik Blockeel and Wannes Meert.

Copyright (c) 2018 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio
import websockets
import json
import time
from collections import defaultdict
import random

logger = logging.getLogger(__name__)
games = {}
agentclass = None


class Board:
    def __init__(self, nb_rows=0, nb_cols=0):

        self.board = [None] * ((2 * nb_rows - 1) * (2 * nb_cols - 1))
        self.dimension = (nb_rows * 2 - 1, nb_cols * 2 - 1)
        self.available_moves = self.free_lines()
        self.boxes = [0, 0]  # Player 2, Player 1

    def max_points(self):
        return (self.dimension[0] * self.dimension[1]) - self.boxes[0] - self.boxes[1]

    def free_lines(self):
        moves, i = [], 1
        for cel in self.board[1::2]:
            if not cel:
                moves.append((i // self.dimension[1], i % self.dimension[1]))
            i += 2
        return moves

    def fill_line(self, x, y, player):
        self.board[self.position_on_board(x, y)] = True
        self.available_moves.remove((x, y))
        return self.calc_score_for_set_line(x, y, player)

    def score(self, x, y, player):
        self.boxes[player - 1] += 1
        self.board[self.position_on_board(x, y - 1)] = player

    def position_on_board(self, x, y):

        return x * self.dimension[1] + y

    def calc_score_for_set_line(self, x, y, player):
        playagain = False
        pos = self.position_on_board(x, y)
        pos_1_to_left = self.position_on_board((x - 1), y)
        pos_1_to_right = self.position_on_board((x + 1), y)
        pos_2_to_left = self.position_on_board((x - 2), y)
        pos_2_to_right = self.position_on_board((x + 2), y)
        if x % 2 == 1:
            if y - 2 >= 0 and self.board[pos] and self.board[pos - 2] and self.board[pos_1_to_left - 1] and self.board[
                pos_1_to_right - 1]:
                playagain = True
                self.score(x, y, player)
            if (y + 2 < self.dimension[1]) and self.board[pos] and self.board[pos + 2] and self.board[
                pos_1_to_left + 1] and self.board[pos_1_to_right + 1]:
                playagain = True
                self.score(x, y, player)
        else:
            if x - 2 >= 0 and self.board[pos] and self.board[pos_2_to_left] and self.board[pos_1_to_left - 1] and \
                    self.board[pos_1_to_left + 1]:
                playagain = True
                self.score(x, y, player)
            if (x + 2 < self.dimension[0]) and self.board[pos] and self.board[pos_2_to_right] and self.board[
                pos_1_to_right - 1] and self.board[pos_1_to_right + 1]:
                playagain = True
                self.score(x, y, player)
        return playagain

    def copy(self):
        csb = Board()
        csb.dimension = self.dimension
        # shallow copies
        csb.board = self.board[:]
        csb.available_moves = self.available_moves[:]
        csb.boxes = self.boxes[:]
        return csb

    def __eq__(self, other):
        return self.board == other.board


class AlphaBeta:
    def __init__(self):
        self.max = sys.maxsize
        self.maxstates = []
        self.minstates = []
        self.visited_states = 0

    def add(self, x):
        self.visited_states = self.visited_states + x

    def alphabeta(self, board, depth, player):
        self.maxstates = []
        self.minstates = []
        self.visited_states = 0

        return self.maximax(board, depth, player, -self.max, self.max)

    def done(self, move):
        return move[0], move[1], 1337

    def maximax(self, board, depth, player, alpha, beta, move=(0, 0)):
        if depth == 0:
            return self.done(move)
        score = board.boxes[player - 1]
        for x, y in board.free_lines():
            current = board.copy()
            for (a1, b1) in self.maxstates:
                if a1 == current:
                    self.add(1)
                    return x, y, b1
            if current.fill_line(x, y, player):
                (a, b, current_score) = self.maximax(current, depth - 1, player, alpha, beta, (x, y))
            else:
                (a, b, current_score) = self.minimin(current, depth - 1, player, alpha, beta, (x, y))
            self.maxstates.append((current, current_score))
            if current_score > score:
                move = (x, y)
                score = current_score
            alpha = self.bigger(score, alpha)
            # print (score)
            if beta <= alpha:
                break
        return move[0], move[1], score

    def minimin(self, board, depth, player, alpha, beta, move=(0, 0)):
        if depth == 0:
            return self.done(move)
        score = board.max_points()
        for x, y in board.free_lines():
            current = board.copy()
            for (a1, b1) in self.minstates:
                if a1 == current:
                    self.add(1)
                    return x, y, b1
            if current.fill_line(x, y, player):
                (a, b, current_score) = self.minimin(current, depth - 1, player, alpha, beta, (x, y))
            else:
                (a, b, current_score) = self.maximax(current, depth - 1, player, alpha, beta, (x, y))
            self.minstates.append((current, current_score))
            if current_score < score:
                move = (x, y)
                score = current_score
            beta = self.smaller(beta, score)
            if beta <= alpha:
                break
            # print (score)
        return move[0], move[1], score

    def bigger(self, a, b):
        if a > b:
            return a
        else:
            return b

    def smaller(self, a, b):
        if a < b:
            return a
        else:
            return b

    def swap_player(self, player):
        if player == 1:
            return 2
        else:
            return 1


class DotsAndBoxesAgent:
    """Example Dots and Boxes agent implementation base class.
    It returns a random next move.

    A DotsAndBoxesAgent object should implement the following methods:
    - __init__
    - add_player
    - register_action
    - next_action
    - end_game

    This class does not necessarily use the best data structures for the
    approach you want to use.
    """

    def __init__(self, player, nb_rows, nb_cols, timelimit):
        """Create Dots and Boxes agent.

        :param player: Player number, 1 or 2
        :param nb_rows: Rows in grid
        :param nb_cols: Columns in grid
        :param timelimit: Maximum time allowed to send a next action.
        """
        self.player = {player}
        self.timelimit = timelimit
        self.ended = False
        self.board = Board(nb_rows + 1, nb_cols + 1)
        self.odds = []
        self.evens = []
        self.times_for_move = []

        i = 0
        while i < 100:
            if i % 2 == 0:
                self.evens.append(i)
            else:
                self.odds.append(i)
            i += 1

    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)

    def register_action(self, row, column, orientation, player):
        """Register action played in game.

        :param row:
        :param column:
        :param orientation: "v" or "h"
        :param player: 1 or 2
        """
        if orientation == "h":
            a = self.evens[row]
            b = self.odds[column]
        else:
            a = self.odds[row]
            b = self.evens[column]
        self.board.fill_line(a, b, player)

    def next_action(self):
        """Return the next action this agent wants to perform.

        In this example, the function implements a random move. Replace this
        function with your own approach.

        :return: (row, column, orientation)
        """
        ab = AlphaBeta()
        start_time = time.time()

        free_lines = self.board.free_lines()
        if len(free_lines) == 0:
            # Board full
            return None

        (a, b, score) = ab.alphabeta(self.board, depth=4, player=list(self.player)[0])

        if a % 2 == 0:
            x = self.odds.index(b)
            y = self.evens.index(a)
            elapsed_time = time.time() - start_time
            self.times_for_move.append(elapsed_time)
            return y, x, "h"
        else:
            y = self.odds.index(a)
            x = self.evens.index(b)
            elapsed_time = time.time() - start_time
            self.times_for_move.append(elapsed_time)
            return y, x, "v"

    def end_game(self):
        time = 0
        for t in self.times_for_move:
            time += t
        print("avg time v2 =", int(time) / len(self.times_for_move))
        self.ended = True


## MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    game = None
    # msg = await websocket.recv()
    try:
        async for msg in websocket:
            logger.info("< {}".format(msg))
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            game = msg["game"]
            answer = None
            if msg["type"] == "start":
                # Initialize game
                if msg["game"] in games:
                    games[msg["game"]].add_player(msg["player"])
                else:
                    nb_rows, nb_cols = msg["grid"]
                    games[msg["game"]] = agentclass(msg["player"],
                                                    nb_rows,
                                                    nb_cols,
                                                    msg["timelimit"])
                if msg["player"] == 1:
                    # Start the game
                    nm = games[game].next_action()
                    print('nm = {}'.format(nm))
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    r, c, o = nm
                    answer = {
                        'type': 'action',
                        'location': [r, c],
                        'orientation': o
                    }
                else:
                    # Wait for the opponent
                    answer = None

            elif msg["type"] == "action":
                # An action has been played
                r, c = msg["location"]
                o = msg["orientation"]
                games[game].register_action(r, c, o, msg["player"])
                if msg["nextplayer"] in games[game].player:
                    # Compute your move
                    nm = games[game].next_action()
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    nr, nc, no = nm
                    answer = {
                        'type': 'action',
                        'location': [nr, nc],
                        'orientation': no
                    }
                else:
                    answer = None

            elif msg["type"] == "end":
                # End the game
                games[msg["game"]].end_game()
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                print(answer)
                await websocket.send(json.dumps(answer))
                logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


## COMMAND LINE INTERFACE

def main(argv=None):
    global agentclass
    parser = argparse.ArgumentParser(description='Start agent to play Dots and Boxes')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentclass = DotsAndBoxesAgent
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
