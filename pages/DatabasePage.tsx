import React from 'react';

const DatabasePage: React.FC = () => {
  const logs = [
     { id: 'LOG-001', timestamp: '2025-05-12 08:00:00', level: 'INFO', message: 'System initialization complete.' },
     { id: 'LOG-002', timestamp: '2025-05-12 08:00:05', level: 'WARN', message: 'Unauthorized access attempt blocked.' },
     { id: 'LOG-003', timestamp: '2025-05-12 08:01:20', level: 'INFO', message: 'Neural link established with central core.' },
     { id: 'LOG-004', timestamp: '2025-05-12 08:05:00', level: 'ERROR', message: 'Packet loss detected in sector 7. Rerouting...' },
     { id: 'LOG-005', timestamp: '2025-05-12 08:15:33', level: 'INFO', message: 'User interaction session started.' },
     { id: 'LOG-006', timestamp: '2025-05-12 08:22:10', level: 'INFO', message: 'Memory buffer optimization: OK.' },
  ];

  const assets = [
    { name: 'Core_Module_v2.dat', size: '2.4 GB', status: 'Active' },
    { name: 'Persona_Matrix.enc', size: '512 MB', status: 'Locked' },
    { name: 'History_Logs.db', size: '120 MB', status: 'Archived' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 py-12 min-h-[80vh]">
      <h1 className="text-2xl md:text-4xl font-['Press_Start_2P'] text-red-600 mb-8 border-b-2 border-red-900 pb-4 flex items-center gap-4">
        <i className="fas fa-database"></i> SYSTEM DATABASE
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Logs Table */}
        <div className="lg:col-span-2">
          <h3 className="text-xl font-['JetBrains_Mono'] text-white mb-4 font-bold flex items-center gap-2">
            <i className="fas fa-list-alt text-red-500"></i> SYSTEM LOGS
          </h3>
          <div className="bg-black/90 border border-red-900 rounded p-6 font-['JetBrains_Mono'] shadow-[0_0_20px_rgba(139,0,0,0.3)] overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm md:text-base">
              <thead>
                <tr className="text-red-500 border-b border-red-800">
                  <th className="p-3">ID</th>
                  <th className="p-3">TIMESTAMP</th>
                  <th className="p-3">LEVEL</th>
                  <th className="p-3">MESSAGE</th>
                </tr>
              </thead>
              <tbody>
                {logs.map(log => (
                  <tr key={log.id} className="border-b border-red-900/30 hover:bg-red-900/10 transition-colors">
                    <td className="p-3 text-gray-500 font-bold">{log.id}</td>
                    <td className="p-3 text-gray-400">{log.timestamp}</td>
                    <td className={`p-3 font-bold ${log.level === 'ERROR' ? 'text-red-500 animate-pulse' : log.level === 'WARN' ? 'text-yellow-500' : 'text-green-500'}`}>{log.level}</td>
                    <td className="p-3 text-gray-300">{log.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Sidebar Stats */}
        <div className="space-y-8">
          {/* Assets Panel */}
          <div>
             <h3 className="text-xl font-['JetBrains_Mono'] text-white mb-4 font-bold flex items-center gap-2">
               <i className="fas fa-file-code text-red-500"></i> ENCRYPTED ASSETS
             </h3>
             <div className="bg-gray-900/50 border border-red-900 p-4 rounded">
                <ul className="space-y-3">
                  {assets.map((asset, idx) => (
                    <li key={idx} className="flex items-center justify-between bg-black p-3 rounded border border-gray-800 hover:border-red-500 transition-colors group">
                      <div className="flex items-center gap-3">
                        <i className="fas fa-file-zipper text-gray-500 group-hover:text-red-400"></i>
                        <span className="font-['JetBrains_Mono'] text-sm">{asset.name}</span>
                      </div>
                      <span className="text-xs text-gray-500">{asset.size}</span>
                    </li>
                  ))}
                </ul>
             </div>
          </div>

          {/* Storage Visualization */}
          <div>
            <h3 className="text-xl font-['JetBrains_Mono'] text-white mb-4 font-bold flex items-center gap-2">
               <i className="fas fa-hard-drive text-red-500"></i> STORAGE
             </h3>
             <div className="bg-black p-4 border border-red-900 rounded text-center">
               <div className="w-32 h-32 rounded-full border-8 border-gray-800 border-t-red-600 border-r-red-600 mx-auto mb-4 animate-spin duration-[3000ms]"></div>
               <p className="font-['Press_Start_2P'] text-red-500">78% FULL</p>
               <p className="text-xs text-gray-500 mt-2">Optimization Recommended</p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatabasePage;