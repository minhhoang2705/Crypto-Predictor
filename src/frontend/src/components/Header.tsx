import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import RefreshIcon from '@mui/icons-material/Refresh';
import { Box } from '@mui/material';

function Header() {
  const handleRefresh = () => {
    // Implement refresh logic
    window.location.reload();
  };

  return (
    <AppBar position="static" sx={{ mb: 2 }}>
      <Toolbar>
        <ShowChartIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Crypto Price Prediction
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2" color="inherit">
            Last Updated: {new Date().toLocaleTimeString()}
          </Typography>
          <IconButton color="inherit" onClick={handleRefresh}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header; 