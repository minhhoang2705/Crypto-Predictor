import axios, { AxiosInstance, AxiosRequestConfig, InternalAxiosRequestConfig } from 'axios';
const rateLimit = require('axios-rate-limit');

// Define types for rate limit options
interface RateLimitOptions {
  maxRequests: number;
  perMilliseconds: number;
}

class ApiService {
  private api: AxiosInstance;
  private requestCount: number = 0;
  private lastRequestTime: number = 0;

  constructor() {
    // Create axios instance with base configuration
    this.api = axios.create({
      baseURL: process.env.REACT_APP_API_URL,
      timeout: parseInt(process.env.REACT_APP_API_TIMEOUT || '30000'),
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.REACT_APP_API_KEY,
      },
    });

    // Add rate limiting
    this.api = rateLimit(this.api, {
      maxRequests: parseInt(process.env.REACT_APP_API_RATE_LIMIT || '60'),
      perMilliseconds: parseInt(process.env.REACT_APP_API_RATE_WINDOW || '60000'),
    } as RateLimitOptions);

    // Add request interceptor
    this.api.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Check rate limiting
        const now = Date.now();
        if (now - this.lastRequestTime < 1000) {
          this.requestCount++;
          if (this.requestCount > parseInt(process.env.REACT_APP_API_RATE_LIMIT || '60')) {
            return Promise.reject(new Error('Rate limit exceeded'));
          }
        } else {
          this.requestCount = 1;
          this.lastRequestTime = now;
        }

        // Add security headers
        if (config.headers) {
          config.headers['X-Request-Time'] = new Date().toISOString();
          config.headers['X-Client-Version'] = process.env.REACT_APP_VERSION || '1.0.0';
        }

        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Add response interceptor
    this.api.interceptors.response.use(
      (response) => {
        // Validate response
        if (!response.data) {
          return Promise.reject(new Error('Invalid response'));
        }
        return response;
      },
      (error) => {
        // Handle errors
        if (error.response) {
          switch (error.response.status) {
            case 401:
              console.error('Unauthorized access');
              break;
            case 403:
              console.error('Forbidden access');
              break;
            case 429:
              console.error('Rate limit exceeded');
              break;
            default:
              console.error('API error:', error.response.status);
              break;
          }
        }
        return Promise.reject(error);
      }
    );
  }

  // API methods
  async getPrediction(data: any) {
    try {
      const response = await this.api.post('/predict', data);
      return response.data;
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  }

  async getMarketData(symbol: string, interval: string) {
    try {
      const response = await this.api.get(`/market-data/${symbol}`, {
        params: { interval },
      });
      return response.data;
    } catch (error) {
      console.error('Market data error:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService(); 