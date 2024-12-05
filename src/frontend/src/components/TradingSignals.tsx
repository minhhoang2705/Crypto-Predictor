import React from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
} from '@mui/material';

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'buy' | 'sell' | 'neutral';
  description: string;
}

const mockIndicators: TechnicalIndicator[] = [
  {
    name: 'RSI (14)',
    value: 65.5,
    signal: 'neutral',
    description: 'Momentum is balanced',
  },
  {
    name: 'MACD',
    value: 245.8,
    signal: 'buy',
    description: 'Bullish crossover detected',
  },
  {
    name: 'Moving Average (50)',
    value: 48250,
    signal: 'buy',
    description: 'Price above MA50',
  },
  {
    name: 'Bollinger Bands',
    value: 0.85,
    signal: 'sell',
    description: 'Price near upper band',
  },
];

function TradingSignals() {
  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'buy':
        return 'success';
      case 'sell':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Technical Indicators
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Indicator</TableCell>
              <TableCell align="right">Value</TableCell>
              <TableCell align="center">Signal</TableCell>
              <TableCell>Analysis</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {mockIndicators.map((indicator) => (
              <TableRow key={indicator.name}>
                <TableCell component="th" scope="row">
                  {indicator.name}
                </TableCell>
                <TableCell align="right">{indicator.value}</TableCell>
                <TableCell align="center">
                  <Chip
                    label={indicator.signal.toUpperCase()}
                    color={getSignalColor(indicator.signal) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>{indicator.description}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Box mt={3}>
        <Typography variant="h6" gutterBottom>
          Trading Recommendation
        </Typography>
        <Paper sx={{ p: 2 }}>
          <Typography variant="body1" gutterBottom>
            Overall Signal: <Chip label="STRONG BUY" color="success" />
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Based on the technical analysis, the market shows bullish momentum with strong buying pressure.
            The MACD indicates a potential upward trend, supported by price action above key moving averages.
            Consider opening long positions with strict stop-loss orders.
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
}

export default TradingSignals; 