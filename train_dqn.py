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
            reward += 100  # Winning the game
        else:
            reward -= 30  # Losing the game
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
                reward += 20  # Increased reward for becoming a king

            # Moving towards opponent's side
            if (turn == RED and row_end < row_start) or (turn == WHITE and row_end > row_start):
                reward += 5  # Reward for advancing a piece

            # Encourage control of central squares
            if row_end in [3, 4] and col_end in [3, 4]:
                reward += 3  # Reward for controlling central squares

            # Penalize for losing a piece (if the opponent can capture it in the next move)
            opponent_turn = WHITE if turn == RED else RED
            opponent_actions = get_valid_actions(board, opponent_turn)
            for opponent_action in opponent_actions:
                _, _, opp_row_end, opp_col_end = opponent_action
                if (opp_row_end, opp_col_end) == (row_end, col_end):
                    reward -= 10  # Penalty for putting the piece in danger

    # Small positive reward for each move to encourage exploration
    reward += 0.1

    return reward

# Function to save the model
def save_model(agent, filename="model.save/dqn_model.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(agent.model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Function to load the model
def load_model(agent, filename="model.save/dqn_model.pth"):
    if os.path.exists(filename):
        agent.model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
    else:
        print(f"No saved model found at {filename}")

# Training loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Flag to visualize the game")
    parser.add_argument("--load", action="store_true", help="Flag to load a pre-trained model")
    args = parser.parse_args()

    agent = DQNAgent(state_size=64, action_size=64)

    # Load model if flag is set
    if args.load:
        load_model(agent)

    episodes = 1000

    # Pygame window setup
    WIN = None
    if args.visualize:
        WIN = pygame.display.set_mode((SQUARE_SIZE * 8, SQUARE_SIZE * 8))
        pygame.display.set_caption('Checkers AI Training')

    for episode in range(episodes):
        board = Board()
        board.reset()  # Ensure the board is reset at the beginning of each episode
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

            # Adjust action size to match the number of valid actions
            action_idx = agent.get_action(state, len(valid_actions))
            if action_idx >= len(valid_actions):
                action_idx = random.choice(range(len(valid_actions)))

            action = valid_actions[action_idx]
            row_start, col_start, row_end, col_end = action
            piece = board.get_piece(row_start, col_start)

            # Check for capture and remove the captured piece explicitly
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

            # Visualize the move in Pygame if the flag is set
            if args.visualize:
                draw_board(WIN, board)
                pygame.time.delay(1)  # Reduced delay to make the moves faster

            next_state = get_state_representation(board)
            done = board.game_over()
            reward = calculate_reward(board, action, turn, done)
            if turn == RED:
                total_reward_red += reward
            else:
                total_reward_white += reward

            agent.remember(state, action_idx, reward, next_state, done)

            # Debugging information
            #print(f"Action taken: {action}, Reward received: {reward}")

            # Train the agent with replay
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.replay()

            state = next_state
            turn = WHITE if turn == RED else RED

        agent.update_epsilon()
        print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {agent.epsilon}, Total Reward (RED): {total_reward_red}, Total Reward (WHITE): {total_reward_white}")

    # Save the trained model
    save_model(agent)

    if args.visualize:
        pygame.quit()
