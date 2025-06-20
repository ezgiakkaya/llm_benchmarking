#!/usr/bin/env python3
"""
MongoDB Migration Script: Local to Atlas
Migrates questions and responses from local MongoDB to MongoDB Atlas
"""

import os
from pymongo import MongoClient
from datetime import datetime
import json

def migrate_database():
    """Migrate data from local MongoDB to Atlas"""
    
    # Configuration
    LOCAL_URI = "mongodb://localhost:27017/"
    ATLAS_URI = input("Enter your Atlas connection string: ").strip()
    
    if not ATLAS_URI:
        print("❌ Atlas connection string is required!")
        return False
    
    # Replace placeholder password
    if "<db_password>" in ATLAS_URI:
        password = input("Enter your database password: ").strip()
        ATLAS_URI = ATLAS_URI.replace("<db_password>", password)
    
    DATABASE_NAME = "comp430_benchmark"
    
    try:
        print("🔄 Connecting to local MongoDB...")
        local_client = MongoClient(LOCAL_URI, serverSelectionTimeoutMS=5000)
        local_db = local_client[DATABASE_NAME]
        
        print("🔄 Connecting to MongoDB Atlas...")
        atlas_client = MongoClient(ATLAS_URI, serverSelectionTimeoutMS=10000)
        atlas_db = atlas_client[DATABASE_NAME]
        
        # Test connections
        local_client.admin.command('ping')
        atlas_client.admin.command('ping')
        print("✅ Both connections successful!")
        
        # Get collections
        collections_to_migrate = ["questions", "responses"]
        migration_summary = {}
        
        for collection_name in collections_to_migrate:
            print(f"\n📦 Migrating collection: {collection_name}")
            
            # Get local collection
            local_collection = local_db[collection_name]
            atlas_collection = atlas_db[collection_name]
            
            # Count documents in local
            local_count = local_collection.count_documents({})
            print(f"   📊 Local documents: {local_count}")
            
            if local_count == 0:
                print(f"   ⚠️  No documents to migrate in {collection_name}")
                migration_summary[collection_name] = {"migrated": 0, "skipped": 0, "errors": 0}
                continue
            
            # Get all documents from local
            documents = list(local_collection.find())
            
            migrated = 0
            skipped = 0
            errors = 0
            
            for doc in documents:
                try:
                    # Check if document already exists in Atlas
                    existing = atlas_collection.find_one({"_id": doc["_id"]})
                    
                    if existing:
                        print(f"   ⏭️  Skipping existing document: {doc.get('_id', 'unknown')}")
                        skipped += 1
                    else:
                        # Insert document to Atlas
                        atlas_collection.insert_one(doc)
                        migrated += 1
                        if migrated % 10 == 0:
                            print(f"   📈 Migrated {migrated} documents...")
                
                except Exception as e:
                    print(f"   ❌ Error migrating document {doc.get('_id', 'unknown')}: {str(e)}")
                    errors += 1
            
            migration_summary[collection_name] = {
                "migrated": migrated,
                "skipped": skipped,
                "errors": errors
            }
            
            print(f"   ✅ {collection_name}: {migrated} migrated, {skipped} skipped, {errors} errors")
        
        # Final summary
        print("\n" + "="*50)
        print("🎉 MIGRATION COMPLETE!")
        print("="*50)
        
        total_migrated = sum(summary["migrated"] for summary in migration_summary.values())
        total_skipped = sum(summary["skipped"] for summary in migration_summary.values())
        total_errors = sum(summary["errors"] for summary in migration_summary.values())
        
        print(f"📊 Total migrated: {total_migrated}")
        print(f"⏭️  Total skipped: {total_skipped}")
        print(f"❌ Total errors: {total_errors}")
        
        # Verify migration
        print("\n🔍 Verifying migration...")
        for collection_name in collections_to_migrate:
            atlas_count = atlas_db[collection_name].count_documents({})
            print(f"   📦 {collection_name} in Atlas: {atlas_count} documents")
        
        print("\n✅ Migration verification complete!")
        print(f"🔗 Your Atlas connection string: {ATLAS_URI.replace(password, '*' * len(password)) if 'password' in locals() else ATLAS_URI}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        return False
    
    finally:
        # Close connections
        try:
            local_client.close()
            atlas_client.close()
        except:
            pass

def check_local_data():
    """Check what data exists in local MongoDB"""
    try:
        print("🔍 Checking local MongoDB data...")
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["comp430_benchmark"]
        
        client.admin.command('ping')
        print("✅ Connected to local MongoDB")
        
        collections = ["questions", "responses"]
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"   📦 {collection_name}: {count} documents")
            
            if count > 0:
                # Show sample document
                sample = collection.find_one()
                print(f"   📄 Sample {collection_name[:-1]}:")
                if collection_name == "questions":
                    print(f"      ID: {sample.get('q_id', 'N/A')}")
                    print(f"      Type: {sample.get('q_type', 'N/A')}")
                    print(f"      Text: {sample.get('q_text', 'N/A')[:50]}...")
                else:
                    print(f"      Question ID: {sample.get('question_id', 'N/A')}")
                    print(f"      Model: {sample.get('model_name', 'N/A')}")
                    print(f"      Timestamp: {sample.get('timestamp', 'N/A')}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Cannot connect to local MongoDB: {str(e)}")
        print("💡 Make sure MongoDB is running locally: brew services start mongodb-community")
        return False

if __name__ == "__main__":
    print("🚀 MongoDB Migration Tool: Local → Atlas")
    print("="*50)
    
    # Check local data first
    if not check_local_data():
        print("\n❌ Cannot proceed without local MongoDB connection")
        exit(1)
    
    print("\n" + "="*50)
    proceed = input("Do you want to proceed with migration? (y/N): ").strip().lower()
    
    if proceed in ['y', 'yes']:
        success = migrate_database()
        if success:
            print("\n🎉 Migration completed successfully!")
            print("💡 You can now add the Atlas connection string to your Kinsta environment variables")
        else:
            print("\n❌ Migration failed. Please check the errors above.")
    else:
        print("Migration cancelled.") 