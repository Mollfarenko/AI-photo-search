from storage.chroma_store import get_chroma_client, get_collection
import json
import sys

def print_help():
    """Display available commands"""
    print("""
╔══════════════════════════════════════════════════════════╗
║              ChromaDB Inspector Commands                 ║
╚══════════════════════════════════════════════════════════╝

  count                    - Show total number of embeddings
  peek [n]                 - Show first n items (default 5)
  get <id>                 - Get specific embedding by ID
  find key=value           - Search by metadata
  list                     - List all IDs
  stats                    - Show collection statistics
  delete_all               - Delete ALL embeddings (requires confirmation)
  delete where key=value   - Delete embeddings matching condition
  delete id <id>           - Delete specific embedding by ID
  help                     - Show this help
  exit/quit                - Exit inspector

Examples:
  peek 10
  get photo_123
  find year=2023
  delete where time_of_day=morning
  delete id photo_123
""")

def print_stats(collection):
    """Print collection statistics"""
    try:
        count = collection.count()
        print(f"\n{'='*50}")
        print(f"Collection: {collection.name}")
        print(f"Total embeddings: {count}")
        
        if count > 0:
            # Sample to get metadata keys
            sample = collection.peek(limit=1)
            if sample and sample.get("metadatas"):
                meta_keys = list(sample["metadatas"][0].keys())
                print(f"Metadata fields: {', '.join(meta_keys)}")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"Error getting stats: {e}")

def main():
    try:
        client = get_chroma_client()
        collection = get_collection(client)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("     ChromaDB Inspector CLI")
    print("="*50)
    print_stats(collection)
    print_help()
    
    while True:
        try:
            cmd = input("\033[1;36mchroma>\033[0m ").strip()  # Cyan prompt
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting inspector. Goodbye!")
            break
        
        if not cmd:
            continue
        
        # Parse command
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if command in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            
            elif command == "help":
                print_help()
            
            elif command == "count":
                count = collection.count()
                print(f"\n✓ Total embeddings: {count}\n")
            
            elif command == "stats":
                print_stats(collection)
            
            elif command == "list":
                result = collection.get()
                ids = result.get("ids", [])
                print(f"\n{'='*50}")
                print(f"Total IDs: {len(ids)}")
                print(f"{'='*50}")
                for i, id in enumerate(ids, 1):
                    print(f"{i:4d}. {id}")
                print()
            
            elif command == "peek":
                n = int(args) if args else 5
                result = collection.peek(limit=n)
                
                if not result or not result.get("ids"):
                    print("\n⚠ Collection is empty\n")
                    continue
                
                print(f"\n{'='*50}")
                print(f"Showing first {len(result['ids'])} embeddings:")
                print(f"{'='*50}\n")
                
                for i, (id, meta) in enumerate(zip(result["ids"], result["metadatas"]), 1):
                    print(f"{i}. ID: \033[1m{id}\033[0m")
                    print(f"   Metadata:")
                    print("   " + json.dumps(meta, indent=3).replace("\n", "\n   "))
                    print()
            
            elif command == "get":
                if not args:
                    print("\n⚠ Usage: get <id>\n")
                    continue
                
                result = collection.get(ids=[args])
                
                if not result or not result.get("ids"):
                    print(f"\n⚠ No embedding found with id: {args}\n")
                    continue
                
                print(f"\n{'='*50}")
                print(f"Embedding ID: \033[1m{args}\033[0m")
                print(f"{'='*50}")
                print(json.dumps(result, indent=2, default=str))
                print()
            
            elif command == "find":
                if not args or "=" not in args:
                    print("\n⚠ Usage: find key=value\n")
                    continue
                
                key, value = args.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                try:
                    if value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        value = float(value)
                except:
                    pass
                
                result = collection.get(where={key: value})
                count = len(result.get("ids", []))
                
                print(f"\n{'='*50}")
                print(f"Found {count} embeddings where {key}={value}")
                print(f"{'='*50}\n")
                
                if count > 0:
                    for i, (id, meta) in enumerate(zip(result["ids"], result["metadatas"]), 1):
                        print(f"{i}. {id}")
                        print("   " + json.dumps(meta, indent=3).replace("\n", "\n   "))
                        print()
            
            elif command == "delete_all":
                count = collection.count()
                print(f"\n⚠ WARNING: This will delete ALL {count} embeddings!")
                confirm = input("Type 'DELETE ALL' to confirm: ")
                
                if confirm == "DELETE ALL":
                    all_ids = collection.get()["ids"]
                    collection.delete(ids=all_ids)
                    print("\n✓ Collection wiped.\n")
                else:
                    print("\n✗ Cancelled.\n")
            
            elif command == "delete":
                if args.startswith("where "):
                    # delete where key=value
                    where_clause = args[6:]  # Remove "where "
                    if "=" not in where_clause:
                        print("\n⚠ Usage: delete where key=value\n")
                        continue
                    
                    key, value = where_clause.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert value
                    try:
                        if value.isdigit():
                            value = int(value)
                    except:
                        pass
                    
                    # Check count first
                    to_delete = collection.get(where={key: value})
                    count = len(to_delete.get("ids", []))
                    
                    if count == 0:
                        print(f"\n⚠ No embeddings found where {key}={value}\n")
                        continue
                    
                    print(f"\nFound {count} embeddings to delete where {key}={value}")
                    confirm = input("Type YES to confirm: ")
                    
                    if confirm == "YES":
                        collection.delete(where={key: value})
                        print(f"\n✓ Deleted {count} embeddings.\n")
                    else:
                        print("\n✗ Cancelled.\n")
                
                elif args.startswith("id "):
                    # delete id <id>
                    id_to_delete = args[3:].strip()
                    
                    # Check if exists
                    result = collection.get(ids=[id_to_delete])
                    if not result.get("ids"):
                        print(f"\n⚠ No embedding found with id: {id_to_delete}\n")
                        continue
                    
                    confirm = input(f"Delete embedding {id_to_delete}? (yes/no): ")
                    if confirm.lower() == "yes":
                        collection.delete(ids=[id_to_delete])
                        print(f"\n✓ Deleted {id_to_delete}\n")
                    else:
                        print("\n✗ Cancelled.\n")
                else:
                    print("\n⚠ Usage: delete where key=value OR delete id <id>\n")
            
            else:
                print(f"\n⚠ Unknown command: '{command}'. Type 'help' for available commands.\n")
        
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
