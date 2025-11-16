"""Test the trained model API endpoint."""
import requests
import json
import time

def test_move_endpoint():
    # Give server time to start
    time.sleep(2)
    
    try:
        resp = requests.post(
            'http://localhost:5058/move',
            json={'pgn': 'e2e4 e7e5', 'timeleft': 5000},
            timeout=5
        )
        
        print(f"Status: {resp.status_code}")
        data = resp.json()
        
        print(f"Move: {data.get('move')}")
        print(f"Error: {data.get('error')}")
        print(f"Time taken: {data.get('time_taken', 0):.2f}ms")
        
        if data.get('move_probs'):
            moves_list = sorted(
                data['move_probs'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print("\nTop 5 moves by probability:")
            for move, prob in moves_list:
                print(f"  {move}: {prob:.4f}")
        
        print("\n✅ API test successful!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    test_move_endpoint()
