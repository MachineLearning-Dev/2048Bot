import sys
import time
from playwright.sync_api import sync_playwright
from bot_core import setup_browser, play_game_loop

def run_target_bot():
    target = 8192
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            print("Invalid target tile. Using default 8192.")
            
    print(f"Starting Target Mode (Target Tile: {target})")
    
    with sync_playwright() as p:
        browser, context, page = setup_browser(p)
        
        attempt = 1
        while True:
            print(f"\n--- Attempt {attempt} ---")
            score, max_tile = play_game_loop(page)
            
            if max_tile >= target:
                print(f"\nSUCCESS! Target {target} reached (Hit {max_tile})!")
                input("Press Enter to close the browser...")
                break
            
            print(f"Target not reached. (Max tile: {max_tile}). Restarting...")
            try:
                page.locator(".retry-button").click(timeout=3000)
                time.sleep(1) # wait for reset
            except:
                print("Could not find retry button. Please check browser.")
                break
                
            attempt += 1
            
        browser.close()

if __name__ == "__main__":
    run_target_bot()
