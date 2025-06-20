# 🚀 Deployment Guide

## Overview
This guide explains how to deploy the COMP430 LLM Benchmark application to Railway using multiple deployment methods.

## Deployment Files Created

### 1. **nixpacks.toml** (Primary Method)
- Configures Nixpacks build system for Railway
- Specifies Python dependencies and start command
- Automatically handles environment setup

### 2. **Procfile** (Alternative Method)
- Simple process file for Railway
- Specifies web process with Streamlit start command

### 3. **Dockerfile** (Container Method)
- Complete Docker containerization
- Includes all system dependencies
- Alternative deployment approach

### 4. **railway.json** (Railway Configuration)
- Railway-specific deployment settings
- Health check configuration
- Restart policies

## Environment Variables Required

Set these in Railway's environment variables section:

```bash
# Required: MongoDB connection string
MONGODB_URI=mongodb://your-connection-string

# Optional: Database name (default: comp430_benchmark)
DATABASE_NAME=comp430_benchmark

# Optional: Connection timeout in milliseconds (default: 5000)
MONGODB_TIMEOUT=5000

# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Groq API
GROQ_API_KEY=your_groq_api_key

# Pinecone (for RAG)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Optional: Custom MongoDB URI
MONGODB_URI=mongodb://localhost:27017/comp430_benchmark
```

## Deployment Steps

### Method 1: Nixpacks (Recommended)

1. **Connect GitHub Repository**
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure Environment Variables**
   - Add all required environment variables
   - Ensure MongoDB URI points to a cloud database (MongoDB Atlas recommended)

3. **Deploy**
   - Railway will automatically detect the `nixpacks.toml`
   - Build process will install dependencies
   - Application will start using the specified command

### Method 2: Docker

1. **Enable Docker Build**
   - Railway will automatically detect the `Dockerfile`
   - Build process will create container image
   - Deploy using containerized application

### Method 3: Manual Configuration

1. **Set Build Command**
   ```
   pip install -r requirements.txt
   ```

2. **Set Start Command**
   ```
   streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
   ```

## Database Setup

### MongoDB Configuration

The app now supports flexible MongoDB configuration for different deployment scenarios.

### Deployment Scenarios

#### 1. Local Development
```bash
export MONGODB_URI="mongodb://localhost:27017/"
```

#### 2. MongoDB Atlas (Cloud)
```bash
export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/"
```

#### 3. Railway Deployment
```bash
# Railway will provide these automatically if you add MongoDB service
export MONGODB_URI="mongodb://mongo:27017/"
```

#### 4. Docker Deployment
```bash
# If using Docker Compose with MongoDB service
export MONGODB_URI="mongodb://mongodb:27017/"
```

### Railway Deployment Steps

1. **Push your code to GitHub**
   ```bash
   git push origin deployment-setup
   ```

2. **Create Railway Project**
   - Go to [Railway](https://railway.app)
   - Connect your GitHub repository
   - Select the `deployment-setup` branch

3. **Add MongoDB Service**
   - In Railway dashboard, click "New Service"
   - Select "Database" → "MongoDB"
   - Railway will automatically set `MONGODB_URI` environment variable

4. **Set Environment Variables**
   - Go to your app service settings
   - Add environment variables:
     ```
     OPENAI_API_KEY=your_openai_key
     GROQ_API_KEY=your_groq_key
     ANTHROPIC_API_KEY=your_anthropic_key
     PINECONE_API_KEY=your_pinecone_key
     ```

5. **Deploy**
   - Railway will automatically deploy when you push changes
   - The app will start in demo mode if MongoDB is not available
   - Once MongoDB is connected, full functionality will be available

### Demo Mode

If MongoDB is not available, the app will run in **demo mode**:
- ✅ LLM testing still works
- ✅ RAG functionality available
- ❌ Question upload disabled
- ❌ Results dashboard limited
- ⚠️  Data not persisted

### Health Checks

The app includes database connection monitoring:
- Green status: Database connected
- Yellow/Red status: Demo mode (database not available)
- Sidebar shows connection details

### Troubleshooting

#### Connection Timeout
- Increase `MONGODB_TIMEOUT` environment variable
- Check network connectivity
- Verify MongoDB service is running

#### Authentication Failed
- Verify username/password in connection string
- Check database user permissions
- Ensure IP whitelist includes your deployment IP

#### Service Discovery
- For Docker: Use service names (e.g., `mongodb:27017`)
- For Railway: Use provided connection strings
- For local: Use `localhost:27017`

### Docker Deployment

The updated Dockerfile includes all necessary dependencies. To deploy:

```bash
# Build image
docker build -t comp430-llm-benchmark .

# Run with environment variables
docker run -d \
  -p 8501:8501 \
  -e MONGODB_URI="mongodb://your-connection-string" \
  -e OPENAI_API_KEY="your-key" \
  comp430-llm-benchmark
```

The app is now production-ready with proper error handling and flexible database configuration! 🎉

## Performance Considerations

### Resource Allocation
- **Memory**: Minimum 512MB recommended
- **CPU**: 0.5 vCPU minimum
- **Storage**: 1GB for dependencies and data

### Optimization
- Use MongoDB Atlas for better performance
- Consider CDN for static assets
- Monitor Railway usage metrics

## Security Notes

1. **Environment Variables**
   - Never commit API keys to repository
   - Use Railway's secure environment variable storage
   - Rotate keys regularly

2. **Database Security**
   - Use MongoDB Atlas with proper authentication
   - Enable network security rules
   - Regular backups recommended

3. **Application Security**
   - Streamlit has built-in security features
   - Consider additional authentication for production use

## Monitoring

### Railway Dashboard
- Monitor application logs
- Check resource usage
- View deployment status

### Application Health
- Health check endpoint: `/`
- Monitor MongoDB connection
- Check API key validity

## Scaling

### Automatic Scaling
- Railway provides automatic scaling based on traffic
- Monitor usage and adjust resources as needed

### Manual Scaling
- Increase memory/CPU allocation in Railway dashboard
- Consider multiple instances for high traffic

## Support

For deployment issues:
1. Check Railway documentation
2. Review build logs in Railway dashboard
3. Verify environment variables
4. Test locally before deploying 