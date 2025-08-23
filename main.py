#!/usr/bin/env python3
"""
LLM Proxy Server Entry Point
"""

import sys
import logging
from proxy import Proxy

from oproxy.config import PROXY_CONFIG, get_supported_providers


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llm_proxy.log')
        ]
    )

def main():
    """Main entry point"""
    print("🤖 Starting LLM Proxy Server...")

    # Setup logging
    setup_logging(PROXY_CONFIG["log_level"])
    logger = logging.getLogger(__name__)

    # Print supported providers
    providers = get_supported_providers()
    logger.info("Supported providers:")
    for name, info in providers.items():
        status = "✅" if info["configured"] else "❌"
        logger.info(f"  {status} {name}: {info['base_url']}")

    # Create and start proxy
    try:
        print(f"\n🚀 LLM Proxy Server starting on {PROXY_CONFIG['host']}:{PROXY_CONFIG['port']}")
        print("📡 Proxy is ready to route requests!")
        print("\nPress Ctrl+C to stop the server\n")

        # Use proxy.py command line interface for compatibility
        import subprocess

        # Build command line arguments
        cmd = [
            'python', '-m', 'proxy',
            '--hostname', PROXY_CONFIG['host'],
            '--port', str(PROXY_CONFIG['port']),
            '--plugins', 'oproxy.plugins.LLMProxyPlugin',
            '--log-level', PROXY_CONFIG['log_level'],
            '--enable-reverse-proxy', 
            '--rewrite-host-header',
            # '--log-file', 'llm_proxy.log',
            # '--enable-dashboard',
        ]

        try:
            # Run proxy process
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down LLM Proxy Server...")

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down LLM Proxy Server...")
    except Exception as e:
        logger.error(f"Failed to start proxy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python main.py")
        print("Starts the LLM Proxy Server.")
        print("Ensure required environment variables are set for desired providers.")
    else:
        main()