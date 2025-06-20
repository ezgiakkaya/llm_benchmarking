#!/usr/bin/env python3
"""
Atlas Connection Test Script
"""

from pymongo import MongoClient
import sys

def test_atlas_connection():
    print("🧪 MongoDB Atlas Connection Test")
    print("="*40)
    
    # Get connection details
    username = input("Enter Atlas username (default: akkayaezgi21): ").strip() or "akkayaezgi21"
    password = input("Enter Atlas password: ").strip()
    cluster = input("Enter cluster address (default: cluster0.fyjnnog.mongodb.net): ").strip() or "cluster0.fyjnnog.mongodb.net"
    
    # Test different connection string formats
    connection_strings = [
        f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0",
        f"mongodb+srv://{username}:{password}@{cluster}/comp430_benchmark?retryWrites=true&w=majority",
        f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority"
    ]
    
    for i, uri in enumerate(connection_strings, 1):
        print(f"\n🔗 Testing connection string #{i}...")
        print(f"   URI: mongodb+srv://{username}:***@{cluster}/...")
        
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=10000)
            
            # Test connection
            client.admin.command('ping')
            print("   ✅ Connection successful!")
            
            # Test database access
            db = client['comp430_benchmark']
            db.test_collection.insert_one({"test": "connection"})
            db.test_collection.delete_one({"test": "connection"})
            print("   ✅ Database write/read successful!")
            
            # List databases
            databases = client.list_database_names()
            print(f"   📊 Available databases: {databases}")
            
            client.close()
            
            # Save working connection string
            with open('.env.atlas', 'w') as f:
                f.write(f"MONGODB_URI={uri}\n")
            print(f"   💾 Working connection string saved to .env.atlas")
            
            return uri
            
        except Exception as e:
            print(f"   ❌ Connection failed: {str(e)}")
            continue
    
    print("\n❌ All connection attempts failed!")
    print("\n🔧 Troubleshooting steps:")
    print("1. Check your Atlas dashboard:")
    print("   - Database Access: Ensure user exists with correct password")
    print("   - Network Access: Add your IP address (0.0.0.0/0 for testing)")
    print("2. Verify cluster name and connection string")
    print("3. Check user permissions (Read and write to any database)")
    
    return None

if __name__ == "__main__":
    working_uri = test_atlas_connection()
    
    if working_uri:
        print("\n🎉 Atlas connection working!")
        print("💡 You can now run the migration with the working connection string")
    else:
        print("\n❌ Please fix Atlas configuration and try again") 