import time
from playwright.sync_api import sync_playwright
from bot_core import setup_browser, play_game_loop

def run_normal_bot():
    print("Starting Playwright...")
    with sync_playwright() as p:
        print("Launching Chromium...")
        browser, context, page = setup_browser(p)
        
        print("Starting game...")
        play_game_loop(page)
        
        input("\nGame finished! Press Enter to close the browser...")
        browser.close()

if __name__ == "__main__":
    run_normal_bot()
