import random
import torch
import pygame
import argparse
import os
from checkers.board import Board
from checkers.constants import RED, WHITE, SQUARE_SIZE
from dqn_agent import DQNAgent

# Pygame setup
def draw_board(win, board):
    board.draw(win)
    pygame.display.update()

# Helper functions for state representation and valid actions
def get_state_representation(board):
    state = []
    for row in board.board:
        for cell in row:
            if cell == 0:
                state.append(0)
            elif cell.color == WHITE and not cell.king:
                state.append(1)  # White piece
            elif cell.color == WHITE and cell.king:
                state.append(2)  # White king
            elif cell.color == RED and not cell.king:
                state.append(-1)  # Red piece
            elif cell.color == RED and cell.king:
                state.append(-2)  # Red king
    return state

def get_valid_actions(board, turn):
    valid_actions = []
    for row in range(8):
        for col in range(8):
            piece = board.get_piece(row, col)
            if piece != 0 and piece.color == turn:
                valid_moves = board.get_valid_moves(piece)
                for move in valid_moves:
                    end_row, end_col = move
                    valid_actions.append((row, col, end_row, end_col))
    return valid_actions

# Reward function
def calculate_reward(board, action, turn, done):
    row_start, col_start, row_end, col_end = action
    piece = board.get_piece(row_start, col_start)

    # Initialize reward
    reward = 0.0

    # Check if the game is won or lost
    if done:
        if board.get_winner() == turn:
            reward += 50  # Winning the game
        else:
            reward -= 50  # Losing the game
    else:
        # Ensure that piece is valid
        if piece != 0:
            # Capture opponent's piece
            valid_moves = board.get_valid_moves(piece)
            if (row_end, col_end) in valid_moves:
                skipped = valid_moves.get((row_end, col_end))
                if skipped:
                    reward += 30  # Increased reward for capturing a piece
                    # Remove the captured piece from the board
                    if isinstance(skipped, list):
                        for skip in skipped:
                            board.remove([skip])
                    else:
                        board.remove([skipped])

            # Becoming a king
            if not piece.king and (row_end == 0 or row_end == 7):
                reward += 10  # Increased reward for becoming a king

            # Penalize for losing a piece (if the opponent can capture it in the next move)
            opponent_turn = WHITE if turn == RED else RED
            opponent_actions = get_valid_actions(board, opponent_turn)
            for opponent_action in opponent_actions:
                _, _, opp_row_end, opp_col_end = opponent_action
                if (opp_row_end, opp_col_end) == (row_end, col_end):
                    reward -= 5  # Penalty for putting the piece in danger

    # Small positive reward for each move to encourage exploration
    reward += 0.1

    return reward

# Function to load the model
def load_model(agent, filename="model.save/dqn_model.pth"):
    if os.path.exists(filename):
        agent.model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
    else:
        print(f"No saved model found at {filename}")

# Testing loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test the trained model")
    parser.add_argument("--visualize", action="store_true", help="Flag to visualize the games during testing")
    args = parser.parse_args()

    # Initialize the DQN Agent
    agent = DQNAgent(state_size=64, action_size=64)

    # Load the trained model
    load_model(agent, filename="model.save/dqn_model.pth")

    episodes = args.episodes

    # Pygame window setup
    WIN = None
    if args.visualize:
        WIN = pygame.display.set_mode((SQUARE_SIZE * 8, SQUARE_SIZE * 8))
        pygame.display.set_caption('Checkers AI Testing')

    for episode in range(episodes):
        board = Board()
        board.reset()
        state = get_state_representation(board)
        done = False
        turn = RED
        total_reward_red = 0
        total_reward_white = 0

        while not done:
            valid_actions = get_valid_actions(board, turn)
            if len(valid_actions) == 0:
                done = True
                continue

            action_idx = agent.get_action(state, len(valid_actions))
            if action_idx >= len(valid_actions):
                action_idx = random.choice(range(len(valid_actions)))

            action = valid_actions[action_idx]
            row_start, col_start, row_end, col_end = action
            piece = board.get_piece(row_start, col_start)

            # Check for capture
            valid_moves = board.get_valid_moves(piece)
            if (row_end, col_end) in valid_moves:
                skipped = valid_moves.get((row_end, col_end))
                if skipped:
                    if isinstance(skipped, list):
                        for skip in skipped:
                            board.remove([skip])
                    else:
                        board.remove([skipped])

            board.move(piece, row_end, col_end)

            if args.visualize:
                draw_board(WIN, board)
                pygame.time.delay(200)

            next_state = get_state_representation(board)
            done = board.game_over()
            reward = calculate_reward(board, action, turn, done)

            if turn == RED:
                total_reward_red += reward
            else:
                total_reward_white += reward

            state = next_state
            turn = WHITE if turn == RED else RED

        print(f"Episode {episode + 1}/{episodes} completed. Total Reward (RED): {total_reward_red}, Total Reward (WHITE): {total_reward_white}")

    if args.visualize:
        pygame.quit()
