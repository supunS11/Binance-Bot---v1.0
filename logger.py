from colorama import Fore, init
import logging
import os

init(autoreset=True)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_info(message):
    print(Fore.GREEN + message)
    logging.info(message)

def log_warning(message):
    print(Fore.YELLOW + message)
    logging.warning(message)

def log_error(message):
    print(Fore.RED + message)
    logging.error(message)