#!/usr/bin/env python3
"""
Deployment Setup Script
Prepares environment variables for Kinsta deployment
"""

import os

def setup_deployment_env():
    print("🚀 COMP430 LLM Benchmark - Deployment Setup")
    print("="*50)
    

    print("\n📊 MongoDB Atlas Configuration:")
    print("1. Go to your Atlas dashboard")
    print("2. Ensure database user is created with read/write permissions")
    print("3. Add your IP to Network Access (or use 0.0.0.0/0)")
    print("4. Get your connection string from Database → Connect → Drivers")
    
    mongodb_uri = input("\nEnter your MongoDB Atlas connection string: ").strip()
    
    if not mongodb_uri:
        print("❌ MongoDB URI is required for full functionality")
        mongodb_uri = "not-set"
    
    # Get API keys (optional)
    print("\n🔑 API Keys (optional but recommended):")
    openai_key = input("Enter OpenAI API key (or press Enter to skip): ").strip() or "not-set"
    groq_key = input("Enter Groq API key (or press Enter to skip): ").strip() or "not-set"
    

    env_content = f"""# MongoDB Configuration
MONGODB_URI={mongodb_uri}
DATABASE_NAME=comp430_benchmark
MONGODB_TIMEOUT=10000

# API Keys
OPENAI_API_KEY={openai_key}
GROQ_API_KEY={groq_key}

# Production Settings
NODE_ENV=production
"""
    

    with open('.env.kinsta', 'w') as f:
        f.write(env_content)
    
    print("\n✅ Environment configuration saved to .env.kinsta")
    
 
    print("\n🚀 Kinsta Deployment Instructions:")
    print("="*50)
    print("1. Go to your Kinsta application dashboard")
    print("2. Navigate to 'Environment Variables'")
    print("3. Add these variables:")
    print()
    
    for line in env_content.strip().split('\n'):
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            if 'not-set' not in value:
                print(f"   {key} = {value}")
            else:
                print(f"   {key} = (optional - skip if not available)")
    
    print("\n4. Save and redeploy your application")
    print("5. Your app will switch from demo mode to full functionality")
    
  
    if mongodb_uri != "not-set" and "mongodb" in mongodb_uri.lower():
        print(f"\n🧪 Testing MongoDB connection...")
        try:
            from pymongo import MongoClient
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
            client.admin.command('ping')
            print("✅ MongoDB connection successful!")
            
            # Check for existing data
            db = client['comp430_benchmark']
            questions_count = db.questions.count_documents({})
            responses_count = db.responses.count_documents({})
            
            if questions_count > 0 or responses_count > 0:
                print(f"📊 Found existing data: {questions_count} questions, {responses_count} responses")
            else:
                print("📊 Database is empty - ready for new data")
            
            client.close()
            
        except Exception as e:
            print(f"⚠️  MongoDB connection test failed: {str(e)}")
            print("💡 Fix Atlas configuration and test again")
    
    print(f"\n🎉 Setup complete!")
    return True

if __name__ == "__main__":
    setup_deployment_env() 