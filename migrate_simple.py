#!/usr/bin/env python3
"""
Simple MongoDB Migration Script: Local to Atlas
"""

from pymongo import MongoClient
import sys

def migrate_to_atlas():
    # Configuration - EDIT THESE VALUES
    LOCAL_URI = "mongodb://localhost:27017/"
    
    # REPLACE WITH YOUR ACTUAL VALUES:
    ATLAS_USERNAME = "akkayaezgi21"
    ATLAS_PASSWORD = input("Enter your Atlas password: ").strip()
    ATLAS_CLUSTER = "cluster0.fyjnnog.mongodb.net"
    
    ATLAS_URI = f"mongodb+srv://{ATLAS_USERNAME}:{ATLAS_PASSWORD}@{ATLAS_CLUSTER}/?retryWrites=true&w=majority&appName=Cluster0"
    
    DATABASE_NAME = "comp430_benchmark"
    
    try:
        print("🔄 Connecting to local MongoDB...")
        local_client = MongoClient(LOCAL_URI, serverSelectionTimeoutMS=5000)
        local_db = local_client[DATABASE_NAME]
        local_client.admin.command('ping')
        print("✅ Local MongoDB connected")
        
        print("🔄 Connecting to MongoDB Atlas...")
        atlas_client = MongoClient(ATLAS_URI, serverSelectionTimeoutMS=15000)
        atlas_db = atlas_client[DATABASE_NAME]
        atlas_client.admin.command('ping')
        print("✅ Atlas connected")
        
        # Migrate collections
        collections = ["questions", "responses"]
        
        for collection_name in collections:
            print(f"\n📦 Migrating {collection_name}...")
            
            local_collection = local_db[collection_name]
            atlas_collection = atlas_db[collection_name]
            
            # Get all documents
            documents = list(local_collection.find())
            total = len(documents)
            
            if total == 0:
                print(f"   ⚠️  No documents in {collection_name}")
                continue
            
            print(f"   📊 Found {total} documents to migrate")
            
            # Clear existing data in Atlas (optional)
            existing_count = atlas_collection.count_documents({})
            if existing_count > 0:
                print(f"   ⚠️  Atlas already has {existing_count} documents in {collection_name}")
                overwrite = input(f"   🤔 Overwrite existing {collection_name}? (y/N): ").strip().lower()
                if overwrite in ['y', 'yes']:
                    atlas_collection.delete_many({})
                    print(f"   🗑️  Cleared existing {collection_name}")
            
            # Insert documents
            if documents:
                try:
                    result = atlas_collection.insert_many(documents, ordered=False)
                    print(f"   ✅ Successfully migrated {len(result.inserted_ids)} documents")
                except Exception as e:
                    print(f"   ❌ Error during bulk insert: {str(e)}")
                    # Try individual inserts
                    success = 0
                    for doc in documents:
                        try:
                            atlas_collection.insert_one(doc)
                            success += 1
                        except:
                            pass
                    print(f"   ✅ Individually inserted {success} documents")
        
        # Verify migration
        print("\n🔍 Verification:")
        for collection_name in collections:
            local_count = local_db[collection_name].count_documents({})
            atlas_count = atlas_db[collection_name].count_documents({})
            print(f"   📦 {collection_name}: Local={local_count}, Atlas={atlas_count}")
        
        print(f"\n✅ Migration complete!")
        print(f"🔗 Your Atlas URI: mongodb+srv://{ATLAS_USERNAME}:***@{ATLAS_CLUSTER}/?retryWrites=true&w=majority&appName=Cluster0")
        
        # Save connection string for deployment
        with open('.env.atlas', 'w') as f:
            f.write(f"MONGODB_URI={ATLAS_URI}\n")
        print("💾 Connection string saved to .env.atlas")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        return False
    
    finally:
        try:
            local_client.close()
            atlas_client.close()
        except:
            pass

if __name__ == "__main__":
    print("🚀 Simple MongoDB Migration: Local → Atlas")
    print("="*50)
    
    success = migrate_to_atlas()
    if success:
        print("\n🎉 Ready for deployment!")
        print("💡 Next steps:")
        print("   1. Copy the MONGODB_URI from .env.atlas")
        print("   2. Add it to your Kinsta environment variables")
        print("   3. Redeploy your application")
    else:
        print("\n❌ Migration failed") 