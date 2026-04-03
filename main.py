import sys
import os
import logging
from dragon_vision_core import DragonVisionCore
from dragon_hilo_farmer import start_farming

# إعداد السجلات الرئيسية
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MASTER] %(levelname)s: %(message)s'
)

def display_menu():
    print("=" * 60)
    print("      🔴 DRAGON403 MASTER COMMAND CENTER 🔴")
    print("      Investigator: SULAIMAN_ALSHAMMARI")
    print("=" * 60)
    print("\n[1] 👁️ DRAGON VISION (Capture & Test)")
    print("[2] 💰 HILO COIN FARMER (Auto-Clicker Lite)")
    print("[3] 🛡️ TACTICAL MONITORING (Continuous Search)")
    print("[4] 🔐 FORENSIC REPORT GENERATOR (Manifest)")
    print("[0] 🛑 EXIT SYSTEM")
    print("-" * 60)

def main():
    while True:
        display_menu()
        choice = input("\n>>> SELECT MISSION: ")

        if choice == "1":
            print("\n📸 Initiating Dragon Vision Test...")
            dragon = DragonVisionCore()
            path = dragon.capture_full_screen()
            print(f"✅ Full screen capture saved: {path}")

        elif choice == "2":
            print("\n🚀 Initiating HILO Coin Farmer...")
            start_farming()

        elif choice == "3":
            target = input(">>> Enter Target Image Name (e.g., 'claim.png'): ")
            interval = float(input(">>> Enter Check Interval (seconds): "))
            print(f"\n👀 Starting Tactical Monitoring for: {target}")
            dragon = DragonVisionCore()
            dragon.monitor_and_react(target, interval=interval)

        elif choice == "4":
            print("\n📋 Generating Forensic Manifest (Running Manifest Builder)...")
            # استدعاء ملف السكريبت القديم حقك
            os.system("python dragon403_manifest_generator.py")

        elif choice == "0":
            print("\n🛑 SHUTTING DOWN DRAGON403... STAY SAFE.")
            break

        else:
            print("\n⚠️ INVALID CHOICE. TRY AGAIN.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 SYSTEM INTERRUPTED.")
        sys.exit(0)
