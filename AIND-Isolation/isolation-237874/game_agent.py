#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 05:28:48 2018

@author: uttam
"""

"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """This heurisitc calculates the left blank space on the gameboard.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
            return float('-inf')

    if game.is_winner(player):
        return float('inf')
    
    blank_spaces = len(game.get_blank_spaces())
    total_spaces = game.width * game.height
    return float((blank_spaces/total_spaces)*100)

def custom_score_2(game, player):
    """his score is removes 3 legal_moves if the player is in a corner position, because the chance to win is always worse if you are in the corners.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
            return float('-inf')

    if game.is_winner(player):
        return float('inf')
    
    my_moves = len(game.get_legal_moves(player))
    
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    corners_array = [(0,0), (0, game.height - 1), (game.width - 1, 0), (game.width - 1, game.height - 1)]
    
    if game.get_player_location(player) in corners_array:
        # downgrade, because of bad position on the gameboard
        my_moves -= 3

    return float(my_moves - opponent_moves)    
        
def custom_score_3(game, player):
    """this score is equal to the difference of available moves for the player and twice for the oppenent - so it is more aggressive, because of the multiplier
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(my_moves - 2 * opponent_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            #return the best move for this moment
            return self.best_move

        # Return the best move from the last completed search iteration
        return self.best_move
    
    def is_time_over(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        # take the worst case initially
        best_current_score = float('-inf')
        self.best_current_move = legal_moves[0]
        for m in legal_moves:
            self.is_time_over()
            next_level = game.forecast_move(m)
            # computer starts
            score = self.min_value(next_level, depth - 1)
            if score > best_current_score:
                best_current_score = score
                self.best_current_move = m
        return self.best_current_move
    
    def min_value(self, game, depth):
        if depth == 0:
            return self.score(game, self)
        legal_moves = game.get_legal_moves()
        # start worst case for minimizing level
        best_current_score = float('inf')
        for m in legal_moves:
            self.is_time_over()
            next_level = game.forecast_move(m)
            score = self.max_value(next_level, depth - 1)
            if score < best_current_score:
                best_current_score = score        
        return best_current_score

    def max_value(self, game, depth):
        if depth == 0:
            return self.score(game, self)
        legal_moves = game.get_legal_moves()
        best_current_score = float('-inf')
        for m in legal_moves:
            self.is_time_over()
            next_level = game.forecast_move(m)
            score = self.max_value(next_level, depth - 1)
            if score > best_current_score:
                best_current_score = score        
        return best_current_score

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    def is_time_over(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.best_move = (-1, -1)
        depth = 1
        try:
            while (True):
                move = self.alphabeta(game, depth)
                if move is not (-1, -1):
                    self.best_move = move
                depth += 1
                
                if self.time_left() < self.TIMER_THRESHOLD:
                    return self.best_move
                
        except SearchTimeout:
            #return best move with best possible depth
            return self.best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        self.best_current_move = legal_moves[0]
        for m in legal_moves:
            self.is_time_over()
            next_level = game.forecast_move(m)
            # computer starts
            score = self.min_value(next_level, depth - 1, alpha, beta)
            if score > alpha:
                alpha = score
                self.best_current_move = m
        return self.best_current_move
    
    def min_value(self, game, depth, alpha, beta):
        if depth == 0:
            return self.score(game, self)
        legal_moves = game.get_legal_moves()
        for m in legal_moves:
            self.is_time_over()
            next_level = game.forecast_move(m)
            score = self.max_value(next_level, depth - 1, alpha, beta)
            if score < beta:
                beta = score 
                if beta <= alpha:
                    #we can prune this branch
                    break       
        return beta

    def max_value(self, game, depth, alpha, beta):
        if depth == 0:
            return self.score(game, self)
        legal_moves = game.get_legal_moves()
        for m in legal_moves:
            self.is_time_over()
            next_level = game.forecast_move(m)
            score = self.min_value(next_level, depth - 1, alpha, beta)
            if score > alpha:
                alpha = score 
                if alpha >= beta:
                    #we can prune this branch
                    break       
        return alpha