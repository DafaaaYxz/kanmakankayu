import React, { useEffect, useState } from 'react';
import { useConfig } from '../contexts/ConfigContext';
import { NavLink } from 'react-router-dom';
import { ChatLog } from '../types';

const UserDashboard: React.FC = () => {
  const { currentUser, fetchChatLogs } = useConfig();
  const [history, setHistory] = useState<ChatLog[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
        const logs = await fetchChatLogs();
        setHistory(logs);
        setLoading(false);
    };
    loadData();
  }, [fetchChatLogs]);

  // Group stats
  const totalMessages = history.length;
  const userMessages = history.filter(h => h.role === 'user').length;
  
  if (!currentUser) return null;

  return (
    <div className="min-h-screen bg-black text-gray-200 p-4 md:p-8 font-['JetBrains_Mono']">
       <div className="max-w-6xl mx-auto">
           {/* Welcome Header */}
           <div className="bg-red-900/10 border-l-4 border-red-600 p-6 mb-8 rounded-r">
               <h1 className="text-2xl font-['Press_Start_2P'] text-white mb-2">
                   WELCOME, <span className="text-red-500">{currentUser.username}</span>
               </h1>
               <p className="text-sm text-gray-400">Access Key: <span className="font-mono bg-black px-2 py-1 rounded border border-gray-800">{currentUser.accessKey}</span></p>
           </div>

           {/* Quick Actions */}
           <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
               <div className="bg-gray-900/50 p-6 rounded border border-gray-800 hover:border-red-500 transition-all group">
                   <h3 className="text-lg font-bold text-white mb-4"><i className="fa-solid fa-terminal text-red-500 mr-2"></i> INITIATE INTERFACE</h3>
                   <p className="text-sm text-gray-400 mb-6">Enter the neural network terminal to execute commands and chat.</p>
                   <NavLink to="/terminal" className="inline-block bg-red-600 text-white font-['Press_Start_2P'] text-xs px-6 py-4 rounded shadow-[4px_4px_0_0_#333] hover:translate-y-1 hover:shadow-none transition-all">
                       OPEN TERMINAL
                   </NavLink>
               </div>
               
               <div className="bg-gray-900/50 p-6 rounded border border-gray-800">
                   <h3 className="text-lg font-bold text-white mb-4"><i className="fa-solid fa-chart-line text-blue-500 mr-2"></i> SESSION STATS</h3>
                   <div className="grid grid-cols-2 gap-4">
                       <div className="bg-black p-4 rounded text-center">
                           <div className="text-2xl font-bold text-white">{totalMessages}</div>
                           <div className="text-xs text-gray-500">Total Logs</div>
                       </div>
                       <div className="bg-black p-4 rounded text-center">
                           <div className="text-2xl font-bold text-white">{userMessages}</div>
                           <div className="text-xs text-gray-500">Commands</div>
                       </div>
                   </div>
               </div>
           </div>

           {/* Chat History */}
           <div className="bg-black border border-red-900/30 rounded p-6 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
               <h3 className="text-xl font-['Press_Start_2P'] text-white mb-6 border-b border-gray-800 pb-4">
                   HISTORY LOGS
               </h3>
               
               {loading ? (
                   <div className="text-center py-10 text-gray-500">Decrypting Logs...</div>
               ) : history.length === 0 ? (
                   <div className="text-center py-10 text-gray-500 italic">No operations recorded.</div>
               ) : (
                   <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                       {history.map((log) => (
                           <div key={log.id} className={`flex ${log.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                               <div className={`max-w-[80%] p-3 rounded text-sm ${
                                   log.role === 'user' 
                                   ? 'bg-red-900/20 border border-red-900/50 text-gray-300' 
                                   : 'bg-gray-900 border border-gray-800 text-gray-400'
                               }`}>
                                   <div className="flex justify-between items-center mb-1 gap-4">
                                       <span className="text-[10px] font-bold uppercase">{log.role === 'user' ? 'YOU' : 'SYSTEM'}</span>
                                       <span className="text-[10px] text-gray-600">{new Date(log.createdAt).toLocaleString()}</span>
                                   </div>
                                   <div className="whitespace-pre-wrap">{log.content.substring(0, 300)}{log.content.length > 300 ? '...' : ''}</div>
                               </div>
                           </div>
                       ))}
                   </div>
               )}
           </div>
       </div>
    </div>
  );
};

export default UserDashboard;
