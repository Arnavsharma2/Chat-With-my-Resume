#!/usr/bin/env python3
"""
Demo Chat with Resume - Shows the professional interface without API calls
"""

from colorama import init, Fore, Style, Back

# Initialize colorama for colored terminal output
init(autoreset=True)

def demo_chat():
    """Demo the professional chat interface"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{'CHAT WITH ARNAV SHARMA - PROFESSIONAL AI ASSISTANT':^70}")
    print(f"{Fore.CYAN}{'='*70}")
    print(f"\n{Fore.GREEN}Hello! I'm Arnav Sharma's AI assistant. I can discuss:")
    print(f"{Fore.WHITE}• My current role as Software Engineering Intern at Wefire")
    print(f"{Fore.WHITE}• My technical skills in Python, ML, and AI development")
    print(f"{Fore.WHITE}• My projects including LSTM stock prediction and customer churn analysis")
    print(f"{Fore.WHITE}• My education at Penn State University (Computer Science)")
    print(f"{Fore.WHITE}• My experience with data analysis, web development, and AI applications")
    print(f"\n{Fore.YELLOW}Type 'quit', 'exit', or 'bye' to end the conversation.")
    print(f"{Fore.CYAN}{'='*70}\n")
    
    # Sample professional responses
    sample_responses = [
        "I'm currently a Software Engineering Intern at Wefire, where I develop comprehensive Reddit data analysis and monitoring solutions for financial sentiment tracking. I've built AI-powered tools using Python and Google Gemini API, implementing automated bots with PRAW library and SMTP notification pipelines. What specific aspect of my work at Wefire interests you most?",
        
        "I have extensive experience in machine learning and AI development, with projects including an LSTM neural network for stock return prediction and a customer churn predictor using PyCaret AutoML. I'm particularly passionate about building solutions that solve real-world problems. Which of my technical projects would you like to learn more about?",
        
        "I'm pursuing a Bachelor of Science in Computer Science at Penn State University with a minor in Artificial Intelligence, expected to graduate in 2026. My focus is on AI/ML applications and software development, and I've maintained a strong academic record while gaining practical experience through internships and personal projects. What would you like to know about my educational background?",
        
        "I've developed several full-stack applications, including a PSU Menu Analyzer website with AI-powered nutritional analysis using Google Gemini API, and this very Chat with Resume system using RAG technology. I enjoy working across the entire technology stack and building innovative solutions. What type of projects are you most interested in discussing?"
    ]
    
    response_index = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print(f"\n{Fore.GREEN}Thank you for chatting! Have a great day! 👋")
                break
            
            if not user_input:
                continue
            
            # Show demo response
            print(f"\n{Fore.MAGENTA}Arnav: {Style.RESET_ALL}", end="")
            print(sample_responses[response_index % len(sample_responses)])
            response_index += 1
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.GREEN}Thank you for chatting! Have a great day! 👋")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}")
            print(f"{Fore.YELLOW}Please try again or type 'quit' to exit.\n")

if __name__ == "__main__":
    demo_chat()
