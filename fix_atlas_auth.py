#!/usr/bin/env python3
"""
MongoDB Atlas Authentication Fix Script
This script helps diagnose and fix MongoDB Atlas authentication issues.
"""

import os
import sys
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, OperationFailure
import urllib.parse

def test_connection(uri, timeout=10000):
    """Test MongoDB connection with detailed error reporting."""
    print(f"🔍 Testing connection to: {uri}")
    
    try:
        # Parse the URI to extract components
        if uri.startswith('mongodb+srv://'):
            # Extract username and password
            parts = uri.split('@')
            if len(parts) < 2:
                print("❌ Invalid URI format - missing @ separator")
                return False
                
            auth_part = parts[0].replace('mongodb+srv://', '')
            host_part = parts[1]
            
            if ':' in auth_part:
                username, password = auth_part.split(':', 1)
                print(f"📊 Username: {username}")
                print(f"📊 Password length: {len(password)} characters")
                print(f"📊 Host: {host_part.split('/')[0]}")
            else:
                print("❌ No username:password found in URI")
                return False
        
        # Create client with timeout
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout,
            connectTimeoutMS=timeout,
            socketTimeoutMS=timeout
        )
        
        # Test basic connection
        print("🔄 Testing server connection...")
        client.admin.command('ping')
        print("✅ Server connection successful!")
        
        # Test database access
        print("🔄 Testing database access...")
        db_name = os.getenv("DATABASE_NAME", "comp430_benchmark")
        db = client[db_name]
        
        # Try to list collections
        collections = db.list_collection_names()
        print(f"✅ Database access successful! Found {len(collections)} collections: {collections}")
        
        # Test read/write permissions
        print("🔄 Testing read/write permissions...")
        test_collection = db["_connection_test"]
        
        # Try to insert a test document
        test_doc = {"test": "connection", "timestamp": "2025-01-20"}
        result = test_collection.insert_one(test_doc)
        print(f"✅ Write permission successful! Inserted document with ID: {result.inserted_id}")
        
        # Try to read the document back
        found_doc = test_collection.find_one({"_id": result.inserted_id})
        if found_doc:
            print("✅ Read permission successful!")
        
        # Clean up test document
        test_collection.delete_one({"_id": result.inserted_id})
        print("✅ Delete permission successful!")
        
        client.close()
        return True
        
    except OperationFailure as e:
        if e.code == 8000:
            print("❌ AUTHENTICATION FAILED (Error 8000)")
            print("💡 This usually means:")
            print("   1. Wrong username or password")
            print("   2. Database user doesn't have correct permissions")
            print("   3. Database user isn't configured for the specific database")
            return False
        else:
            print(f"❌ Database operation failed: {e}")
            return False
            
    except ServerSelectionTimeoutError as e:
        print(f"❌ Server selection timeout: {e}")
        print("💡 This usually means:")
        print("   1. Network connectivity issues")
        print("   2. Incorrect hostname in connection string")
        print("   3. IP address not whitelisted in Atlas")
        return False
        
    except ConnectionFailure as e:
        print(f"❌ Connection failure: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def generate_fixed_uri():
    """Generate a properly formatted MongoDB URI."""
    print("\n🔧 ATLAS CONFIGURATION CHECKLIST:")
    print("=" * 50)
    
    print("\n1. DATABASE USER SETUP:")
    print("   - Go to MongoDB Atlas → Database Access")
    print("   - Click 'Add New Database User'")
    print("   - Username: akkayaezgi21")
    print("   - Password: samsun5555")
    print("   - Database User Privileges: 'Atlas Admin' (NOT Read and Write to any database)")
    print("   - Built-in Role: 'Atlas Admin'")
    
    print("\n2. NETWORK ACCESS:")
    print("   - Go to MongoDB Atlas → Network Access")
    print("   - Click 'Add IP Address'")
    print("   - Select 'Allow Access from Anywhere' (0.0.0.0/0)")
    
    print("\n3. CORRECT CONNECTION STRING:")
    print("   - Current format: mongodb+srv://akkayaezgi21:samsun5555@cluster0.fyjnnog.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    print("   - Add database name: mongodb+srv://akkayaezgi21:samsun5555@cluster0.fyjnnog.mongodb.net/comp430_benchmark?retryWrites=true&w=majority&appName=Cluster0")
    
    print("\n4. ENVIRONMENT VARIABLES FOR KINSTA:")
    print("   MONGODB_URI=mongodb+srv://akkayaezgi21:samsun5555@cluster0.fyjnnog.mongodb.net/comp430_benchmark?retryWrites=true&w=majority&appName=Cluster0")
    print("   DATABASE_NAME=comp430_benchmark")
    print("   MONGODB_TIMEOUT=10000")
    
    return "mongodb+srv://akkayaezgi21:samsun5555@cluster0.fyjnnog.mongodb.net/comp430_benchmark?retryWrites=true&w=majority&appName=Cluster0"

def main():
    """Main function to run the diagnosis."""
    print("🚀 MongoDB Atlas Authentication Diagnostic Tool")
    print("=" * 50)
    
    # Get URI from environment or use the one from logs
    uri = os.getenv("MONGODB_URI", "mongodb+srv://akkayaezgi21:samsun5555@cluster0.fyjnnog.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    
    print(f"📊 Current URI: {uri}")
    print(f"📊 Database Name: {os.getenv('DATABASE_NAME', 'comp430_benchmark')}")
    
    # Test current connection
    success = test_connection(uri)
    
    if not success:
        print("\n" + "=" * 50)
        print("🔧 SUGGESTED FIXES:")
        fixed_uri = generate_fixed_uri()
        
        print(f"\n🔄 Testing suggested URI...")
        test_connection(fixed_uri)

if __name__ == "__main__":
    main() 