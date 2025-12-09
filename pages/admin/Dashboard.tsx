import React, { useState, useEffect } from 'react';
import { useConfig } from '../../contexts/ConfigContext';
import { useNavigate } from 'react-router-dom';

const Dashboard: React.FC = () => {
  const { db, isAdmin, logout, addApiKey, removeApiKey, addUser, deleteUser, updateGlobalConfig } = useConfig();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'config' | 'users' | 'keys' | 'html_gen'>('users');

  const [aiName, setAiName] = useState(db.globalConfig.aiName);
  const [devName, setDevName] = useState(db.globalConfig.devName);
  const [avatarPreview, setAvatarPreview] = useState(db.globalConfig.avatarUrl || '');
  
  const [clientName, setClientName] = useState('');
  const [customAiName, setCustomAiName] = useState('');
  const [customDevName, setCustomDevName] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  
  const [newKey, setNewKey] = useState('');
  const [isAddingKey, setIsAddingKey] = useState(false);

  const [htmlAiName, setHtmlAiName] = useState('');
  const [htmlDevName, setHtmlDevName] = useState('');
  const [htmlKey, setHtmlKey] = useState('');

  // Sync state when db changes (Realtime updates)
  useEffect(() => {
    setAiName(db.globalConfig.aiName);
    setDevName(db.globalConfig.devName);
    setAvatarPreview(db.globalConfig.avatarUrl || '');
  }, [db.globalConfig]);

  useEffect(() => {
    if (!isAdmin) {
      navigate('/login');
    }
  }, [isAdmin, navigate]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        setAvatarPreview(base64String);
        updateGlobalConfig({ avatarUrl: base64String });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSaveConfig = () => {
    updateGlobalConfig({ aiName, devName });
    alert('Global System Configuration Updated');
  };

  const handleGenerateUser = (e: React.FormEvent) => {
    e.preventDefault();
    if (!clientName) return;

    setIsGenerating(true);

    setTimeout(async () => {
        const generatedKey = 'ck_' + Math.random().toString(36).substring(2, 8) + Math.random().toString(36).substring(2, 8);
        
        await addUser({
          id: Date.now().toString(), // Will be UUID in Supabase
          username: clientName, 
          accessKey: generatedKey, 
          role: 'user',
          createdAt: new Date().toISOString(),
          profile: {
            fullName: clientName,
            address: 'N/A', street: 'N/A', zipCode: 'N/A', country: 'N/A'
          },
          config: {
            ...db.globalConfig,
            aiName: customAiName || db.globalConfig.aiName, 
            devName: customDevName || db.globalConfig.devName, 
          }
        });

        setClientName('');
        setCustomAiName('');
        setCustomDevName('');
        setIsGenerating(false);
        alert(`Access Key Generated: ${generatedKey}`);
    }, 1000);
  };

  const handleAddKey = async () => {
      if (!newKey.trim()) return;
      setIsAddingKey(true);
      await addApiKey(newKey);
      setNewKey('');
      setIsAddingKey(false);
  };
  
  const generateStandaloneHtml = () => {
    if(!htmlAiName || !htmlDevName || !htmlKey) {
        alert("Please fill all fields");
        return;
    }

    const template = `<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${htmlAiName}</title>
  <style>
    body { background: #0a0a0a; color: #fff; font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
    .container { text-align: center; }
    h1 { color: #ff0000; }
    .message { margin-top: 20px; border: 1px solid #333; padding: 20px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>${htmlAiName}</h1>
    <p>Developer: ${htmlDevName}</p>
    <div class="message">
      This is a generated standalone template for Key: ${htmlKey}
    </div>
  </div>
</body>
</html>`;

    const blob = new Blob([template], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${htmlAiName.replace(/\s+/g, '_')}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getExpirationStatus = (dateStr: string) => {
    const created = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - created.getTime();
    const diffHours = diffMs / (1000 * 60 * 60);
    const remaining = 24 - diffHours;

    if (remaining <= 0) {
        return <span className="text-red-500 font-bold">EXPIRED</span>;
    }
    return <span className="text-green-500">{Math.floor(remaining)}h {Math.floor((remaining % 1) * 60)}m left</span>;
  };

  return (
    <div className="min-h-screen bg-black text-gray-200 font-['JetBrains_Mono'] p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-10 border-b border-red-900 pb-4">
          <div>
            <h1 className="text-xl md:text-2xl font-['Press_Start_2P'] text-white">
              ADMIN <span className="text-red-600">PANEL</span>
            </h1>
            <p className="text-xs text-gray-500 mt-2">Logged in as {db.users.find(u => u.role === 'admin')?.username}</p>
          </div>
          <button 
            onClick={() => { logout(); navigate('/login'); }}
            className="px-4 py-2 border border-red-600 text-red-500 hover:bg-red-900 hover:text-white transition-colors text-xs font-bold"
          >
            LOGOUT
          </button>
        </div>

        <div className="flex gap-4 mb-8 overflow-x-auto">
          {[
            { id: 'users', label: 'TEMPLATE FACTORY', icon: 'fa-users' },
            { id: 'keys', label: 'API MANAGER', icon: 'fa-key' },
            { id: 'config', label: 'GLOBAL CONFIG', icon: 'fa-gears' },
            { id: 'html_gen', label: 'HTML GENERATOR', icon: 'fa-code' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-6 py-3 border-t-2 text-sm font-bold flex items-center gap-2 transition-all whitespace-nowrap ${
                activeTab === tab.id 
                  ? 'border-red-600 bg-red-900/20 text-white' 
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              <i className={`fa-solid ${tab.icon}`}></i> {tab.label}
            </button>
          ))}
        </div>

        <div className="bg-gray-900/50 border border-red-900 rounded p-6 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
          
          {activeTab === 'users' && (
            <div>
              <h3 className="text-xl text-white font-bold mb-6 border-l-4 border-red-600 pl-3">GENERATE CLIENT ACCESS</h3>
              <form onSubmit={handleGenerateUser} className="bg-black p-6 mb-10 border border-gray-800 rounded relative overflow-hidden">
                <div className="mb-6">
                  <input 
                    placeholder="Client Name (e.g. Budi)" 
                    value={clientName}
                    onChange={e => setClientName(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 p-3 rounded text-sm focus:border-red-500 outline-none text-white placeholder-gray-600"
                    required
                    disabled={isGenerating}
                  />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <input 
                    placeholder="Custom AI Name" 
                    value={customAiName}
                    onChange={e => setCustomAiName(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 p-3 rounded text-sm focus:border-blue-500 outline-none text-white placeholder-gray-600"
                    disabled={isGenerating}
                  />
                   <input 
                    placeholder="Custom Developer Name" 
                    value={customDevName}
                    onChange={e => setCustomDevName(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 p-3 rounded text-sm focus:border-blue-500 outline-none text-white placeholder-gray-600"
                    disabled={isGenerating}
                  />
                </div>
                <button type="submit" disabled={isGenerating} className="w-full bg-red-800 text-white py-3 font-bold font-['Press_Start_2P'] text-xs hover:bg-red-700">
                    {isGenerating ? 'GENERATING...' : 'GENERATE ACCESS KEY'}
                </button>
              </form>

              <h3 className="text-xl text-white font-bold mb-6 border-l-4 border-blue-600 pl-3">ACTIVE SESSIONS</h3>
              <div className="overflow-x-auto rounded border border-gray-800">
                <table className="w-full text-left text-sm">
                  <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
                    <tr>
                      <th className="p-3">Client Name</th>
                      <th className="p-3">Access Key</th>
                      <th className="p-3">Expires In</th>
                      <th className="p-3">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800 bg-black">
                    {db.users.map(user => (
                      <tr key={user.id} className="hover:bg-gray-900/50">
                        <td className="p-3 font-bold text-white">{user.username}</td>
                        <td className="p-3"><code className="bg-gray-800 px-2 py-1 rounded text-red-400 font-bold select-all">{user.accessKey}</code></td>
                        <td className="p-3 font-mono text-xs">{user.role === 'admin' ? 'N/A' : getExpirationStatus(user.createdAt)}</td>
                        <td className="p-3">
                          {user.role !== 'admin' && (
                            <button onClick={() => deleteUser(user.id)} className="text-red-500"><i className="fa-solid fa-trash"></i></button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'keys' && (
            <div>
              <h3 className="text-xl text-white font-bold mb-6 border-l-4 border-red-600 pl-3">GLOBAL API MANAGER</h3>
              <div className="flex gap-2 mb-8">
                <input 
                  value={newKey}
                  onChange={e => setNewKey(e.target.value)}
                  placeholder="Paste new Google GenAI API Key..."
                  className="flex-1 bg-black border border-gray-700 p-3 rounded focus:border-red-500 outline-none text-white"
                  disabled={isAddingKey}
                />
                <button onClick={handleAddKey} disabled={isAddingKey} className="bg-green-700 text-white px-6 font-bold rounded">
                  {isAddingKey ? 'ADDING...' : 'ADD KEY'}
                </button>
              </div>
              <div className="grid gap-3">
                {db.globalConfig.apiKeys.map((key, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-black p-4 border border-gray-800 rounded">
                    <span className="font-mono text-sm text-gray-300">{key.substring(0, 12)}...</span>
                    <button onClick={() => removeApiKey(key)} className="text-red-500">REMOVE</button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'config' && (
             <div>
                  <label className="block text-gray-400 text-xs mb-2">DEFAULT AI NAME</label>
                  <input value={aiName} onChange={(e) => setAiName(e.target.value)} className="w-full bg-black border border-gray-700 p-3 rounded mb-4 text-white"/>
                  <label className="block text-gray-400 text-xs mb-2">DEFAULT DEV NAME</label>
                  <input value={devName} onChange={(e) => setDevName(e.target.value)} className="w-full bg-black border border-gray-700 p-3 rounded mb-4 text-white"/>
                  <button onClick={handleSaveConfig} className="bg-blue-800 text-white px-6 py-3 rounded font-bold w-full">SAVE SETTINGS</button>
             </div>
          )}

           {activeTab === 'html_gen' && (
            <div>
                 <h3 className="text-xl text-white font-bold mb-6 border-l-4 border-green-600 pl-3">HTML GENERATOR</h3>
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <input className="bg-black border border-gray-700 p-3 rounded text-white" placeholder="AI Name" value={htmlAiName} onChange={e => setHtmlAiName(e.target.value)} />
                    <input className="bg-black border border-gray-700 p-3 rounded text-white" placeholder="Dev Name" value={htmlDevName} onChange={e => setHtmlDevName(e.target.value)} />
                    <input className="bg-black border border-gray-700 p-3 rounded text-white md:col-span-2" placeholder="Key" value={htmlKey} onChange={e => setHtmlKey(e.target.value)} />
                 </div>
                 <button onClick={generateStandaloneHtml} className="w-full bg-green-700 text-white font-bold py-3 rounded">DOWNLOAD .HTML</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;