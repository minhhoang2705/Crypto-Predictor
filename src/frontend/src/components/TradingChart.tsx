import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { Box, Typography, CircularProgress, Select, MenuItem, FormControl, InputLabel } from '@mui/material';

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface VolumeData {
  time: number;
  value: number;
  color: string;
}

type TimeFrame = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

function TradingChart() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chart = useRef<any>(null);
  const resizeObserver = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState<TimeFrame>('1h');

  const handleTimeframeChange = (event: any) => {
    setTimeframe(event.target.value as TimeFrame);
  };

  const formatTime = (timestamp: number): number => {
    // Convert milliseconds to seconds and return as number
    return Math.floor(timestamp / 1000);
  };

  const fetchKlines = async (): Promise<ChartData[]> => {
    try {
      console.log('Fetching data from:', `https://api1.binance.com/api/v3/klines?symbol=BTCUSDT&interval=${timeframe}&limit=100`);
      
      const response = await fetch(
        `https://api1.binance.com/api/v3/klines?symbol=BTCUSDT&interval=${timeframe}&limit=100`
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Response Error:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Received data:', data);
      
      if (!Array.isArray(data)) {
        console.error('Invalid data format:', data);
        throw new Error('Invalid data format received from API');
      }
      
      // Sort data by timestamp to ensure ascending order
      const sortedData = [...data].sort((a, b) => a[0] - b[0]);
      
      const formattedData = sortedData.map((item: any) => ({
        time: formatTime(item[0]),
        open: parseFloat(item[1]),
        high: parseFloat(item[2]),
        low: parseFloat(item[3]),
        close: parseFloat(item[4]),
        volume: parseFloat(item[5])
      }));

      console.log('Final formatted data:', formattedData);
      return formattedData;
    } catch (err) {
      console.error('Detailed error:', err);
      throw err;
    }
  };

  useEffect(() => {
    if (chartContainerRef.current) {
      // Create chart
      chart.current = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
          horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
        },
        rightPriceScale: {
          borderColor: 'rgba(197, 203, 206, 0.8)',
          scaleMargins: {
            top: 0.1,
            bottom: 0.3,
          },
        },
        timeScale: {
          borderColor: 'rgba(197, 203, 206, 0.8)',
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time: number) => {
            const date = new Date(time * 1000);
            const hours = date.getHours().toString().padStart(2, '0');
            const minutes = date.getMinutes().toString().padStart(2, '0');
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            const day = date.getDate().toString().padStart(2, '0');
            
            if (['1m', '5m', '15m', '1h', '4h'].includes(timeframe)) {
              return `${month}-${day} ${hours}:${minutes}`;
            }
            return `${month}-${day}`;
          },
        },
      });

      // Add candlestick series
      const candlestickSeries = chart.current.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      // Add volume series
      const volumeSeries = chart.current.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
          top: 0.7,
          bottom: 0,
        },
      });

      // Initial data load
      const loadData = async () => {
        try {
          setLoading(true);
          setError(null);
          const klineData = await fetchKlines();
          
          if (klineData.length === 0) {
            throw new Error('No data received from API');
          }

          // Ensure unique timestamps by adding small increments to duplicates
          const uniqueData = klineData.map((item, index) => ({
            ...item,
            time: item.time + index
          }));

          console.log('Setting candlestick data:', uniqueData);
          candlestickSeries.setData(uniqueData);

          // Set volume data with correct types
          const volumeData: VolumeData[] = uniqueData.map((item) => ({
            time: item.time,
            value: item.volume || 0,
            color: item.close > item.open ? '#26a69a' : '#ef5350'
          }));
          
          console.log('Setting volume data:', volumeData);
          volumeSeries.setData(volumeData);

          // Set visible range
          chart.current.timeScale().fitContent();
        } catch (err: any) {
          console.error('Load data error:', err);
          setError(err.message || 'Failed to load chart data');
        } finally {
          setLoading(false);
        }
      };

      loadData();

      // Handle resize
      const handleResize = () => {
        if (chartContainerRef.current && chart.current) {
          chart.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
          });
        }
      };

      resizeObserver.current = new ResizeObserver(handleResize);
      resizeObserver.current.observe(chartContainerRef.current);

      // Update data every minute
      const interval = setInterval(loadData, 60000);

      return () => {
        if (chart.current) {
          chart.current.remove();
        }
        if (resizeObserver.current) {
          resizeObserver.current.disconnect();
        }
        clearInterval(interval);
      };
    }
  }, [timeframe]);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          BTC/USDT Chart
        </Typography>
        <FormControl variant="outlined" size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Timeframe</InputLabel>
          <Select
            value={timeframe}
            onChange={handleTimeframeChange}
            label="Timeframe"
          >
            <MenuItem value="1m">1 Minute</MenuItem>
            <MenuItem value="5m">5 Minutes</MenuItem>
            <MenuItem value="15m">15 Minutes</MenuItem>
            <MenuItem value="1h">1 Hour</MenuItem>
            <MenuItem value="4h">4 Hours</MenuItem>
            <MenuItem value="1d">1 Day</MenuItem>
          </Select>
        </FormControl>
      </Box>
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1,
          }}
        >
          <CircularProgress />
        </Box>
      )}
      {error && (
        <Typography color="error" align="center" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: 'calc(100% - 40px)',
        }}
      />
    </Box>
  );
}

export default TradingChart; 