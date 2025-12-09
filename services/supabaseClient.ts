import { createClient } from '@supabase/supabase-js';

// GANTI DENGAN URL & ANON KEY DARI DASHBOARD SUPABASE ANDA
// Settings -> API
const SUPABASE_URL = 'https://vugvdkkbluvitjpscylu.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ1Z3Zka2tibHV2aXRqcHNjeWx1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxNTA0NTEsImV4cCI6MjA3ODcyNjQ1MX0.RY0IyPBR812JDO-rsm7UPaINFYbl65crsBH3CLrW8HM';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
