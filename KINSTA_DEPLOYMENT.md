# 🚀 Kinsta Deployment Instructions

## Environment Variables to Add

Go to your Kinsta application dashboard → Environment Variables and add these:

### Required MongoDB Configuration
```
MONGODB_URI=mongodb+srv://akkayaezgi21:samsun5555@cluster0.fyjnnog.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
DATABASE_NAME=comp430_benchmark
MONGODB_TIMEOUT=10000
```

### Optional API Keys (for full LLM functionality)
```
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

### Production Settings
```
NODE_ENV=production
```

## Deployment Steps

1. **Add Environment Variables:**
   - Go to Kinsta dashboard
   - Navigate to your app → Environment Variables
   - Add each variable above
   - Save changes

2. **Redeploy Application:**
   - Kinsta will automatically redeploy
   - Or manually trigger deployment

3. **Verify Deployment:**
   - Visit: https://llmbenchmarking-6ano9.kinsta.app/
   - Should show: ✅ Database: Database connection active
   - No more "demo mode" warnings

## Migration Strategy

### Option 1: Start Fresh (Recommended)
- Deploy with empty Atlas database
- Upload questions through the web interface
- Generate new test results

### Option 2: Migrate Local Data
- Fix Atlas network access (add 0.0.0.0/0 to IP whitelist)
- Run migration script locally
- Then deploy

## Expected Results

After deployment with MongoDB:
- ✅ Database Status: Connected
- ✅ Upload Questions: Fully functional
- ✅ Run Tests: Works with API keys
- ✅ Results Dashboard: Shows real data
- ✅ All features: Operational

## Troubleshooting

If database connection fails:
1. Check Atlas Network Access (allow 0.0.0.0/0)
2. Verify user credentials in Database Access
3. Test connection string locally first
4. Check Kinsta deployment logs

Your app is ready for production! 🎉 