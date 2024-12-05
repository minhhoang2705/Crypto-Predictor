import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Stack,
  Button,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

// Configure the API base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface Prediction {
  price_direction: 'up' | 'down';
  confidence: number;
  target_price: number;
  time_to_target: number;
  current_price: number;
}

function PredictionPanel() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchPrediction = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          // Add your prediction request payload here
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrediction();
    const interval = setInterval(fetchPrediction, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Price Predictions
      </Typography>
      
      {prediction && (
        <Stack spacing={2}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Direction
              </Typography>
              <Box display="flex" alignItems="center" gap={1}>
                {prediction.price_direction === 'up' ? (
                  <Chip
                    icon={<TrendingUpIcon />}
                    label="Bullish"
                    color="success"
                    variant="outlined"
                  />
                ) : (
                  <Chip
                    icon={<TrendingDownIcon />}
                    label="Bearish"
                    color="error"
                    variant="outlined"
                  />
                )}
                <Typography variant="body2">
                  Confidence: {(prediction.confidence * 100).toFixed(2)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Target Price
              </Typography>
              <Typography variant="h4">
                ${prediction.target_price.toLocaleString()}
              </Typography>
              <Box display="flex" alignItems="center" gap={1} mt={1}>
                <AccessTimeIcon fontSize="small" />
                <Typography variant="body2">
                  Expected in {prediction.time_to_target} hours
                </Typography>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Current Price
              </Typography>
              <Typography variant="h5">
                ${prediction.current_price.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {Math.abs(((prediction.target_price - prediction.current_price) / prediction.current_price) * 100).toFixed(2)}% {prediction.price_direction === 'up' ? 'potential gain' : 'potential loss'}
              </Typography>
            </CardContent>
          </Card>
        </Stack>
      )}

      <Button
        variant="contained"
        color="primary"
        fullWidth
        onClick={fetchPrediction}
        sx={{ mt: 2 }}
      >
        Refresh Prediction
      </Button>
    </Box>
  );
}

export default PredictionPanel; 