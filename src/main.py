from database.database import PineconeDatabase
from processors.raw_process import RawProcess
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    print(project_root)
    db = PineconeDatabase()
    ps = RawProcess(project_root, db)
    
    while True:
        command = input("Enter a command: ")
        if command == "exit":
            break
        elif command == "process":
           try:
                ps.run()
           except Exception as e:
                print(f"Error processing raw data: {e}")
        elif command == "query":
            query = input("Enter a query: ")
            print("TBD")
        else:
            print("Invalid command")
    
    