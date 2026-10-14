"""Lesson 3 — Voice in your app.

Voice commands and mouse clicks drive ONE shared game state. Both input paths
call the same `play()` function and both are recorded in the conversation so
the voice agent always knows what happened on the board.
"""
import random

from ..pipeline import Agent, Session
from ..providers.llm import run_tools

WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
NAMES = ["top left", "top middle", "top right", "middle left", "center", "middle right", "bottom left", "bottom middle", "bottom right"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_move",
            "description": "Place the player's X. Squares: 1 top-left, 2 top-middle, 3 top-right, 4 middle-left, 5 center, "
            "6 middle-right, 7 bottom-left, 8 bottom-middle, 9 bottom-right.",
            "parameters": {"type": "object", "properties": {"square": {"type": "integer", "minimum": 1, "maximum": 9}}, "required": ["square"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "new_game", "description": "Clear the board and start a new game.", "parameters": {"type": "object", "properties": {}}},
    },
]


def winner(b):
    for a, c, d in WINS:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return "draw" if " " not in b else None


def minimax(b, player):
    w = winner(b)
    if w:
        return {"O": 1, "X": -1, "draw": 0}[w], None
    best = (-2, None) if player == "O" else (2, None)
    for i in range(9):
        if b[i] == " ":
            b[i] = player
            score, _ = minimax(b, "X" if player == "O" else "O")
            b[i] = " "
            if (player == "O" and score > best[0]) or (player == "X" and score < best[0]):
                best = (score, i)
    return best


class TicTacToe(Agent):
    def __init__(self):
        self.board = [" "] * 9

    @property
    def system(self):
        rows = [" | ".join(c if c != " " else str(i * 3 + j + 1) for j, c in enumerate(self.board[i * 3 : i * 3 + 3])) for i in range(3)]
        return (
            "You are a playful voice opponent in tic-tac-toe. The player is X, you are O.\n"
            "When the player says where to move, call make_move with the square number; the tool also returns your (O) reply move.\n"
            "Then tell the player in ONE short sentence where you moved, using words like 'top left' or 'center' (never numbers), "
            "and react to the game result if it ended. If they ask for a new game call new_game. No markdown.\n"
            f"Current board (numbers are empty squares):\n" + "\n".join(rows)
        )

    def state(self):
        return {"board": self.board, "result": winner(self.board)}

    def play(self, square: int) -> dict:
        i = int(square) - 1
        if winner(self.board):
            return {"error": "The game is over. Ask for a new game."}
        if not 0 <= i < 9 or self.board[i] != " ":
            return {"error": f"Square {square} is not available."}
        self.board[i] = "X"
        result = {"player_move": NAMES[i]}
        if not winner(self.board):
            empty = [k for k in range(9) if self.board[k] == " "]
            # play perfectly most of the time, but slip up sometimes so humans can win
            j = random.choice(empty) if random.random() < 0.3 else minimax(self.board[:], "O")[1]
            self.board[j] = "O"
            result["my_move"] = NAMES[j]
        result["result"] = {"X": "player wins", "O": "you win", "draw": "draw", None: "in progress"}[winner(self.board)]
        return result

    async def on_start(self, s: Session):
        self.board = [" "] * 9
        await s.event("state", **self.state())

    async def respond(self, s: Session, text: str):
        s.history.append({"role": "user", "content": text})

        async def make_move(square: int):
            out = self.play(square)
            await s.event("state", **self.state())
            return out

        async def new_game():
            self.board = [" "] * 9
            await s.event("state", **self.state())
            return {"ok": True}

        async def on_call(call):
            await s.event("tool_call", tool=call["name"], arguments=call["arguments"])

        yield await run_tools(s.llm(), self.system, s.history, TOOLS, {"make_move": make_move, "new_game": new_game}, on_call)

    async def on_action(self, s: Session, data: dict):
        """Mouse clicks: same game logic, recorded in the same conversation."""
        if data.get("new_game"):
            self.board = [" "] * 9
            await s.event("state", **self.state())
            s.history.append({"role": "user", "content": "[clicked New game]"})
            line = "Fresh board. Your move!"
        else:
            out = self.play(data["square"])
            await s.event("state", **self.state())
            s.history.append({"role": "user", "content": f"[clicked square {data['square']}]"})
            if "error" in out:
                line = out["error"]
            elif out["result"] == "player wins":
                line = "You got me! Nice win."
            elif out["result"] == "draw":
                line = f"I'll take {out.get('my_move', 'the last square')}. It's a draw!" if out.get("my_move") else "It's a draw!"
            elif out["result"] == "you win":
                line = f"I'll take {out['my_move']}. Three in a row, I win!"
            else:
                line = f"I'll take {out['my_move']}."
        s.history.append({"role": "assistant", "content": line})
        await s.say(line)
