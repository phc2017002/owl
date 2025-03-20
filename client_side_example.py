import requests
import time
import json

def submit_question(url, question, timeout=120):
    """Submit a question to the API."""
    payload = {
        "question": question,
        "timeout": timeout
    }
    response = requests.post(f"{url}/api/questions", json=payload)
    return response.json()

def check_task_status(url, task_id):
    """Check the status of a task."""
    response = requests.get(f"{url}/api/tasks/{task_id}")
    return response.json()

def main():
    base_url = "http://localhost:8000"
    
    # Submit a question
    question = "Give me analysis of the tesla stock."
    print(f"Submitting question: {question}")
    task = submit_question(base_url, question)
    print(f"Task submitted with ID: {task['task_id']}")
    
    # Poll for task completion
    while True:
        status = check_task_status(base_url, task['task_id'])
        print(f"Task status: {status['status']}")
        
        if status['status'] in ['completed', 'error', 'timeout']:
            break
        
        time.sleep(5)  # Poll every 5 seconds
    
    # Print the final result
    if status['status'] == 'completed':
        print("\nAnswer:")
        print(status['answer'])
        print(f"\nToken count: {status['token_count']}")
    else:
        print(f"\nError: {status['error']}")

if __name__ == "__main__":
    main()