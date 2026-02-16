export default function ChatbotPage() {
    return (
        <div style={{
            padding: 'var(--space-lg)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '80vh',
        }}>
            <div className="glass-card" style={{
                width: '100%',
                maxWidth: '800px',
                height: '600px',
                padding: 0,
                overflow: 'hidden',
                borderRadius: 'var(--radius-lg)',
            }}>
                <iframe
                    id="spai-demo-chatbot-iframe"
                    className="flex h-full w-full"
                    src="https://chatbot.sparkagentai.com/app?t=8cad2075-2014-4dbc-8e20-4d4f29dbf098"
                    style={{
                        border: 'none',
                        width: '100%',
                        height: '100%',
                        borderRadius: 'var(--radius-lg)',
                    }}
                    title="AI Chatbot"
                />
            </div>
        </div>
    );
}
