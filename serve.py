
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os

# Disable MCTS for stability
os.environ['USE_MCTS'] = 'false'

from src.utils import chess_manager
import random
from src import main

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Fallback if move is None: pick a legal move to avoid 500s
    if move is None:
        board = chess_manager.current_context.board if chess_manager.current_context else chess.Board()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return JSONResponse(
                content={
                    "move": None,
                    "move_probs": {},
                    "time_taken": time_taken,
                    "error": "No legal moves available",
                    "logs": logs or [],
                },
                status_code=200,
            )
        move = random.choice(legal_moves)
        # Create uniform probabilities over legal moves
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        # Be lenient: build a uniform distribution if types are off
        board = chess_manager.current_context.board if chess_manager.current_context else chess.Board()
        legal_moves = list(board.legal_moves)
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves} if legal_moves else {}

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move):
            # Try to parse UCI keys if agent returned strings
            try:
                m = chess.Move.from_uci(str(m))
            except Exception:
                continue
        try:
            prob = float(prob)
        except Exception:
            prob = 0.0

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
