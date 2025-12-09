import { createClient } from '@supabase/supabase-js';

// GANTI DENGAN URL & ANON KEY DARI DASHBOARD SUPABASE ANDA
// Settings -> API
const SUPABASE_URL = 'https://loofouwempvwstevseqk.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxvb2ZvdXdlbXB2d3N0ZXZzZXFrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUyNjIxMDksImV4cCI6MjA4MDgzODEwOX0.JWLxaDSCVGulCQZ8yX2I3ONyzbNRdZUQ10EMB6nJoHA';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
