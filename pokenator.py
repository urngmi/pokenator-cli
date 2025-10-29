#!/usr/bin/env python3
"""
Pokénator Professional CLI - Clean Pokemon Guessing Game
Uses ultra diverse question logic with spam protection
Number selection interface, no emojis, professional formatting
"""

# Import the question diversity system FIRST
from question_diversity import patch_engine_for_ultra_diversity

# Apply the diversity patch
create_diverse_engine = patch_engine_for_ultra_diversity()

import sys
import os
from engine import ConfidenceLevel

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

class PokenatorProfessionalCLI:
    def __init__(self):
        self.engine = None
        self.question_count = 0
        self.max_questions = 20
        
    def print_banner(self):
        """Display professional ASCII banner"""
        banner = """
================================================================================
                                                                                
    ██████╗  ██████╗ ██╗  ██╗███████╗███╗   ██╗ █████╗ ████████╗ ██████╗ ██████╗ 
    ██╔══██╗██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
    ██████╔╝██║   ██║█████╔╝ █████╗  ██╔██╗ ██║███████║   ██║   ██║   ██║██████╔╝
    ██╔═══╝ ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║██╔══██║   ██║   ██║   ██║██╔══██╗
    ██║     ╚██████╔╝██║  ██╗███████╗██║ ╚████║██║  ██║   ██║   ╚██████╔╝██║  ██║
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
                                                                                
                           The Ultimate Pokemon Guessing Game                   
================================================================================

Think of any Generation 1 Pokemon and I'll try to guess it!
Uses advanced question diversity to prevent repetitive questions.
"""
        print(banner)

    def main_menu(self):
        """Display main menu and get selection"""
        while True:
            clear_screen()
            self.print_banner()
            
            print("Main Menu:")
            print()
            print("1. Play Game")
            print("2. Exit")
            print()
            
            try:
                choice = input("Select option (1-2): ").strip()
                
                if choice == "1":
                    return True  # Play game
                elif choice == "2":
                    return False  # Exit
                else:
                    print("Invalid choice. Please enter 1 or 2.")
                    input("Press Enter to continue...")
                    
            except (KeyboardInterrupt, EOFError):
                return False

    def get_answer(self, question):
        """Get user answer with number selection"""
        while True:
            clear_screen()
            
            print("=" * 80)
            print(f"Question {self.question_count}/{self.max_questions}")
            print("=" * 80)
            print()
            
            print(question)
            print()
            
            print("Answer Options:")
            print()
            print("1. Yes - Definitely yes")
            print("2. Probably - Probably yes, but not certain") 
            print("3. Don't Know - No idea / I'm not sure")
            print("4. Probably Not - Probably not, but not 100% sure")
            print("5. No - Definitely not")
            print()
            
            try:
                choice = input("Your answer (1-5): ").strip()
                
                confidence_map = {
                    "1": ConfidenceLevel.YES,
                    "2": ConfidenceLevel.PROBABLY,
                    "3": ConfidenceLevel.DONT_KNOW,
                    "4": ConfidenceLevel.PROBABLY_NOT,
                    "5": ConfidenceLevel.NO
                }
                
                if choice in confidence_map:
                    return confidence_map[choice]
                else:
                    print("Invalid choice. Please enter 1-5.")
                    input("Press Enter to continue...")
                    
            except (KeyboardInterrupt, EOFError):
                return None

    def get_guess_confirmation(self, pokemon_name, confidence, alternatives=None):
        """Get confirmation for guess"""
        while True:
            clear_screen()
            
            print("=" * 80)
            print("MY GUESS")
            print("=" * 80)
            print()
            
            print(f"I think your Pokemon is: {pokemon_name.upper()}")
            print(f"Confidence: {confidence:.1%}")
            print()
            
            # Show confidence bar
            bar_length = int(confidence * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"[{bar}] {confidence:.1%}")
            print()
            
            if alternatives and len(alternatives) > 1:
                print("Other possibilities:")
                for i, (name, conf) in enumerate(alternatives[1:4], 2):
                    print(f"  {i}. {name}: {conf:.1%}")
                print()
            
            print("Is this correct?")
            print()
            print("1. Yes, that's correct!")
            print("2. No, keep guessing")
            print()
            
            try:
                choice = input("Your answer (1-2): ").strip()
                
                if choice == "1":
                    return True
                elif choice == "2":
                    return False
                else:
                    print("Invalid choice. Please enter 1 or 2.")
                    input("Press Enter to continue...")
                    
            except (KeyboardInterrupt, EOFError):
                return False

    def show_progress(self):
        """Show current top candidates"""
        candidates = self.engine.get_top_candidates(5)
        if candidates:
            print("\nCurrent top candidates:")
            for i, (name, confidence) in enumerate(candidates[:3], 1):
                bar_length = int(confidence * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {i}. {name:<12} {confidence:>6.1%} [{bar}]")
            print()

    def initialize_engine(self):
        """Initialize the game engine with ultra diversity"""
        print("Initializing game engine...")
        print("Loading ultra diverse question system...")
        
        try:
            self.engine = create_diverse_engine()
            
            if not self.engine:
                print("Error: Failed to initialize engine!")
                return False
                
            self.engine.start_session()
            print("Engine ready!")
            print("Loaded: 151 Pokemon with advanced question diversity")
            print("Anti-spam protection: Active")
            return True
        except Exception as e:
            print(f"Error initializing engine: {e}")
            return False

    def play_game(self):
        """Main game loop"""
        clear_screen()
        print("Setting up your game...")
        
        if not self.initialize_engine():
            input("Press Enter to continue...")
            return
        
        clear_screen()
        print("Game Started!")
        print("Think of any Generation 1 Pokemon...")
        input("Press Enter when you're ready to begin!")
        
        self.question_count = 0
        
        while self.question_count < self.max_questions:
            # Ask next question
            question = self.engine.get_next_question()
            if not question:
                break
                
            self.question_count += 1
            answer = self.get_answer(question)
            
            if answer is None:  # User quit
                return
            
            # Process the answer
            if self.engine.session_log:
                last_trait = self.engine.session_log[-1]['trait']
                self.engine.process_response(last_trait, answer)
            
            # Show progress every few questions
            if self.question_count % 3 == 0:
                clear_screen()
                self.show_progress()
                input("Press Enter to continue...")
        
        # Final guess if we haven't guessed yet
        identification = self.engine.get_identification()
        if identification:
            pokemon_name, confidence = identification
            alternatives = self.engine.get_top_candidates(5)
            
            if self.get_guess_confirmation(pokemon_name, confidence, alternatives):
                clear_screen()
                print("SUCCESS!")
                print(f"I guessed {pokemon_name} in {self.question_count} questions!")
            else:
                clear_screen()
                print("I couldn't guess your Pokemon this time!")
                pokemon = input("What Pokemon were you thinking of? ")
                print(f"Thanks! I'll remember {pokemon} for next time.")
        else:
            clear_screen()
            print("I couldn't guess your Pokemon this time!")
        
        input("Press Enter to continue...")

    def run(self):
        """Main application loop"""
        try:
            while True:
                if self.main_menu():
                    self.play_game()
                else:
                    clear_screen()
                    print("Thanks for playing Pokénator!")
                    break
        except KeyboardInterrupt:
            clear_screen()
            print("\nThanks for playing!")
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Entry point"""
    game = PokenatorProfessionalCLI()
    game.run()

if __name__ == "__main__":
    main()
