import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import DatasetRegistry from './pages/DatasetRegistry';
import FineTuning from './pages/FineTuning';
import ModelRegistry from './pages/ModelRegistry';
import Jobs from './pages/Jobs';

// Create a React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30000,
    },
  },
});

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/datasets" element={<DatasetRegistry />} />
            <Route path="/finetune" element={<FineTuning />} />
            <Route path="/models" element={<ModelRegistry />} />
            <Route path="/jobs" element={<Jobs />} />
            <Route path="*" element={<div className="p-6">Page not found</div>} />
          </Routes>
        </Layout>
      </Router>
    </QueryClientProvider>
  );
};

export default App;
