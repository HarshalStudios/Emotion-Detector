/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Header, NavTab } from './components/Header';
import { Footer } from './components/Footer';
import { HomePage } from './pages/HomePage';
import { AnalyzePage } from './pages/AnalyzePage';
import { HowItWorksPage } from './pages/HowItWorksPage';
import { ResearchPage } from './pages/ResearchPage';
import { HistoryPage, HistoryRecord } from './pages/HistoryPage';
import { apiService } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('analyze');
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);

  useEffect(() => {
    // Real health check on startup
    const runHealthCheck = async () => {
      const result = await apiService.checkHealth();
      setBackendConnected(result.connected);
    };

    runHealthCheck();
    const interval = setInterval(runHealthCheck, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleRecordHistory = (record: {
    timestamp: string;
    prediction: string;
    confidence: number;
    latencyMs: number;
    geometryValid: boolean;
  }) => {
    setHistory((prev) => [
      {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
        ...record,
      },
      ...prev.slice(0, 49), // Keep latest 50
    ]);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans bg-grid-pattern selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Sticky Header */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        backendConnected={backendConnected}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'home' && (
          <HomePage onNavigate={(tab) => setActiveTab(tab)} />
        )}

        {activeTab === 'analyze' && (
          <AnalyzePage onRecordHistory={handleRecordHistory} />
        )}

        {activeTab === 'how-it-works' && (
          <HowItWorksPage />
        )}

        {activeTab === 'research' && (
          <ResearchPage />
        )}

        {activeTab === 'history' && (
          <HistoryPage
            history={history}
            onClearHistory={handleClearHistory}
          />
        )}
      </main>

      {/* Footer */}
      <Footer onNavigate={(tab) => setActiveTab(tab)} />
    </div>
  );
}
