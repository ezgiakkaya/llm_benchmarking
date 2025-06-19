# Railway Deployment Guide

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
# MongoDB Connection
MONGODB_URI=your_mongodb_connection_string

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

### MongoDB Atlas (Recommended for Production)

1. **Create MongoDB Atlas Cluster**
   - Sign up at mongodb.com/atlas
   - Create a free cluster
   - Get connection string

2. **Configure Network Access**
   - Allow access from anywhere (0.0.0.0/0) for Railway
   - Or whitelist Railway's IP ranges

3. **Set Environment Variable**
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/comp430_benchmark
   ```

## Troubleshooting

### Common Issues

1. **Port Configuration**
   - Ensure `$PORT` environment variable is used
   - Streamlit must bind to `0.0.0.0` for external access

2. **Dependencies**
   - Check `requirements.txt` for all necessary packages
   - Some packages may require system dependencies (handled in Dockerfile)

3. **Environment Variables**
   - Verify all required API keys are set
   - Check MongoDB connection string format

4. **Build Failures**
   - Check Railway build logs for specific errors
   - Ensure Python version compatibility (3.11 recommended)

### Debug Commands

```bash
# Check if application starts locally
streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0

# Test environment variables
python -c "import os; print(os.getenv('MONGODB_URI'))"

# Verify dependencies
pip list
```

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