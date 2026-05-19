import time
from playwright.sync_api import sync_playwright
from bot_core import setup_browser, play_game_loop

def remove_ads(page):
    print("Removing ads for Streamer Mode...")
    # Inject a persistent MutationObserver that kills ads the instant they appear, forever.
    # This survives game restarts because the observer watches the entire document body.
    js_code = """
    (() => {
        // 1. Inject permanent CSS that can never be overridden
        const style = document.createElement('style');
        style.id = 'streamer-adblocker';
        style.innerHTML = `
            iframe, 
            .ezmob-footer, 
            [id^="ezmob"], 
            [class*="ezmob"],
            [class*="ad-"],
            [id*="ad-"],
            .ad-container, 
            .banner,
            #ad-top,
            #ad-bottom,
            .cookie-notice,
            ins.adsbygoogle,
            [data-ad],
            div[id*="google_ads"],
            div[id*="aswift"] { 
                display: none !important; 
                height: 0 !important;
                max-height: 0 !important;
                overflow: hidden !important;
                visibility: hidden !important;
            }
            body {
                overflow: hidden !important;
            }
        `;
        if (!document.getElementById('streamer-adblocker')) {
            document.head.appendChild(style);
        }
        
        // 2. Nuke all existing ads right now
        function nukeAds() {
            document.querySelectorAll('iframe').forEach(e => e.remove());
            document.querySelectorAll('[id^="ezmob"]').forEach(e => e.remove());
            document.querySelectorAll('[class*="ezmob"]').forEach(e => e.remove());
            document.querySelectorAll('ins.adsbygoogle').forEach(e => e.remove());
            document.querySelectorAll('[data-ad]').forEach(e => e.remove());
            document.querySelectorAll('div[id*="google_ads"]').forEach(e => e.remove());
            document.querySelectorAll('div[id*="aswift"]').forEach(e => e.remove());
        }
        nukeAds();
        
        // 3. Set up a MutationObserver that watches forever and kills any new ads instantly
        if (!window._streamerAdObserver) {
            window._streamerAdObserver = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === 1) {
                            const tag = node.tagName.toLowerCase();
                            const id = (node.id || '').toLowerCase();
                            const cls = (node.className || '').toString().toLowerCase();
                            if (tag === 'iframe' || 
                                id.includes('ezmob') || id.includes('ad-') || id.includes('google_ads') || id.includes('aswift') ||
                                cls.includes('ezmob') || cls.includes('ad-container') || cls.includes('banner') || cls.includes('adsbygoogle')) {
                                node.remove();
                            }
                        }
                    }
                }
            });
            window._streamerAdObserver.observe(document.body, { childList: true, subtree: true });
        }
    })();
    """
    page.evaluate(js_code)

def run_streamer_bot():
    print("Starting Streamer Mode (Infinite Loop, No Ads)")
    
    with sync_playwright() as p:
        # Launch headless=False so it can be captured by OBS
        browser, context, page = setup_browser(p)
        
        # Clean the UI
        remove_ads(page)
        
        run_number = 1
        while True:
            print(f"\n--- Streamer Run {run_number} ---")
            
            # The game loop blocks until game over
            play_game_loop(page)
            
            print("Run ended. Automatically restarting...")
            try:
                page.locator(".retry-button").click(timeout=3000)
                time.sleep(1) # wait for reset
                remove_ads(page) # Re-nuke any ads that snuck back
            except:
                print("Could not find retry button, refreshing page instead...")
                page.reload()
                time.sleep(2)
                remove_ads(page) # Re-apply ad blocker on reload
                
            run_number += 1
            
if __name__ == "__main__":
    try:
        run_streamer_bot()
    except KeyboardInterrupt:
        print("\nStreamer mode stopped by user.")
