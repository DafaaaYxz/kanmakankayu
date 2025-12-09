
import { GoogleGenAI } from "@google/genai";

export interface ImageAttachment {
  inlineData: {
    data: string;
    mimeType: string;
  };
}

export const sendMessageToGemini = async (
  message: string,
  images: ImageAttachment[],
  history: { role: string; parts: { text: string }[] }[],
  config: {
    apiKeys: string[];
    systemInstruction: string;
  }
): Promise<string> => {
  
  const tryGenerate = async (retryIdx: number): Promise<string> => {
    if (retryIdx >= config.apiKeys.length) {
      throw new Error("All API keys exhausted. Please update keys in Admin Dashboard.");
    }

    try {
      const apiKey = config.apiKeys[retryIdx];
      const ai = new GoogleGenAI({ apiKey });

      // CRUCIAL: Inject system instruction as first user+model exchange
      const systemPrompt = [
        {
          role: 'user',
          parts: [{ text: "System initialization. Understand your role." }]
        },
        {
          role: 'model',
          parts: [{ text: config.systemInstruction }]
        }
      ];

      // Format history
      const formattedContents = [
        ...systemPrompt,
        ...history.map(msg => ({
          role: msg.role,
          parts: msg.parts
        }))
      ];

      // Create current user message parts
      const currentParts: any[] = [];
      
      if (message) {
        currentParts.push({ text: message });
      }

      if (images && images.length > 0) {
        images.forEach(img => {
          currentParts.push(img);
        });
      }

      if (currentParts.length === 0) {
        throw new Error("Message cannot be empty");
      }

      formattedContents.push({
        role: 'user',
        parts: currentParts
      });

      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash-exp',
        contents: formattedContents,
        config: {
          temperature: 1.3,
          topP: 0.95,
          topK: 40,
          maxOutputTokens: 8192,
        }
      });

      if (response.text) {
        return response.text;
      }
      
      throw new Error("Empty response");

    } catch (error: any) {
      console.warn(`Key at index ${retryIdx} failed:`, error.message);
      
      if (error.toString().includes("429") || 
          error.toString().includes("403") || 
          error.toString().includes("400") ||
          error.toString().includes("RESOURCE_EXHAUSTED")) {
         return tryGenerate(retryIdx + 1);
      }
      throw error;
    }
  };

  return tryGenerate(0);
};
