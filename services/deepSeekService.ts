import { PERSONA } from "../constants";

const API_KEY = "sk-51b6904f8d574a1ab2b0a64d3266cee8";
const API_URL = "https://api.deepseek.com/chat/completions";

export const sendMessageToDeepSeek = async (
  message: string,
  history: { role: string; parts: { text: string }[] }[]
): Promise<string> => {
  try {
    // Transform history from Google GenAI format to OpenAI/DeepSeek format
    // App uses: { role: 'user'|'model', parts: [{ text: '...' }] }
    // DeepSeek uses: { role: 'user'|'assistant', content: '...' }
    
    const formattedHistory = history.map(msg => ({
      role: msg.role === 'model' ? 'assistant' : 'user',
      content: msg.parts[0]?.text || ''
    }));

    // DeepSeek chat completion payload
    const payload = {
      model: "deepseek-chat",
      messages: [
        { role: "system", content: PERSONA },
        ...formattedHistory,
        { role: "user", content: message }
      ],
      temperature: 1.3, // High creativity as requested by persona
      stream: false
    };

    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("DeepSeek API Error:", errorText);
      throw new Error(`DeepSeek API returned ${response.status}`);
    }

    const data = await response.json();
    
    // Extract content from OpenAI-compatible response format
    const content = data.choices?.[0]?.message?.content;
    
    if (!content) {
      throw new Error("No content in response");
    }

    return content;

  } catch (error) {
    console.error("DeepSeek Service Error:", error);
    throw new Error("Connection lost to the neural network.");
  }
};