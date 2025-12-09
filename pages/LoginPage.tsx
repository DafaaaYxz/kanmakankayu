import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConfig } from '../contexts/ConfigContext';

const LoginPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'client' | 'dev'>('client');
  const [accessKey, setAccessKey] = useState('');
  const [adminUser, setAdminUser] = useState('');
  const [adminPass, setAdminPass] = useState('');
  const [error, setError] = useState('');
  
  const { loginClient, loginAdmin } = useConfig();
  const navigate = useNavigate();

  const handleClientLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    const result = await loginClient(accessKey);
    if (result.success) {
      navigate('/terminal');
    } else {
      setError(result.message || 'Login failed');
    }
  };

  const handleDevLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    const result = await loginAdmin(adminUser, adminPass);
    if (result.success) {
      navigate('/admin/dashboard');
    } else {
      setError(result.message || 'Login failed');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black relative overflow-hidden px-4">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-red-900/10 via-black to-black"></div>
      <div className="w-full max-w-md bg-black/90 border-2 border-red-900 rounded-lg shadow-[0_0_50px_rgba(139,0,0,0.4)] relative z-10 overflow-hidden">
        
        <div className="bg-red-900/20 p-6 text-center border-b border-red-900">
          <i className="fa-solid fa-shield-halved text-4xl text-red-600 mb-4 animate-pulse"></i>
          <h2 className="text-xl font-['Press_Start_2P'] text-white">SECURE <span className="text-red-600">ACCESS</span></h2>
        </div>

        <div className="flex border-b border-red-900">
          <button onClick={() => { setActiveTab('client'); setError(''); }} className={`flex-1 py-4 text-xs font-bold font-['Press_Start_2P'] ${activeTab === 'client' ? 'bg-red-600 text-white' : 'bg-transparent text-gray-500'}`}>CLIENT KEY</button>
          <button onClick={() => { setActiveTab('dev'); setError(''); }} className={`flex-1 py-4 text-xs font-bold font-['Press_Start_2P'] ${activeTab === 'dev' ? 'bg-red-600 text-white' : 'bg-transparent text-gray-500'}`}>DEV CONSOLE</button>
        </div>

        <div className="p-8">
          {error && <div className="bg-red-900/30 border border-red-600 text-red-200 p-3 mb-6 text-xs">{error}</div>}

          {activeTab === 'client' ? (
            <form onSubmit={handleClientLogin} className="space-y-6">
              <div>
                <label className="block text-red-500 text-xs mb-2 uppercase">Enter 24h Access Key</label>
                <input type="password" value={accessKey} onChange={(e) => setAccessKey(e.target.value)} className="w-full bg-gray-900 border border-red-900/50 p-3 rounded text-white focus:border-red-500 outline-none" placeholder="xxxx-xxxx-xxxx" required />
              </div>
              <button type="submit" className="w-full bg-red-800 text-white font-['Press_Start_2P'] text-xs py-4 rounded hover:bg-red-700">CONNECT</button>
            </form>
          ) : (
            <form onSubmit={handleDevLogin} className="space-y-6">
               <div>
                <label className="block text-red-500 text-xs mb-2 uppercase">Dev Username</label>
                <input type="text" value={adminUser} onChange={(e) => setAdminUser(e.target.value)} className="w-full bg-gray-900 border border-red-900/50 p-3 rounded text-white" placeholder="Username" required />
              </div>
              <div>
                <label className="block text-red-500 text-xs mb-2 uppercase">Admin Key</label>
                <input type="password" value={adminPass} onChange={(e) => setAdminPass(e.target.value)} className="w-full bg-gray-900 border border-red-900/50 p-3 rounded text-white" placeholder="Password" required />
              </div>
              <button type="submit" className="w-full bg-red-800 text-white font-['Press_Start_2P'] text-xs py-4 rounded hover:bg-red-700">AUTHORIZE</button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default LoginPage;