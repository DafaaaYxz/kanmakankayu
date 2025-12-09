import React from 'react';
import ChatInterface from '../components/ChatInterface';

const TerminalPage: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8 mt-10 min-h-[70vh] flex flex-col">
      <div className="text-center mb-8 flex-shrink-0">
          <h2 className="font-['Press_Start_2P'] text-red-500 text-xl md:text-2xl mb-2">INTERFACE</h2>
          <div className="h-1 w-20 bg-red-600 mx-auto"></div>
      </div>
      <div className="flex-1">
        <ChatInterface />
      </div>
    </div>
  );
};

export default TerminalPage;
