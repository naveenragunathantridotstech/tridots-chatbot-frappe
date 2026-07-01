app_name = "tridots_chatbot"
app_title = "Tridots Chatbot"
app_publisher = "Tridots Tech"
app_description = "RAG chatbot widget for tridotstech.com"
app_icon = "octicon octicon-comment-discussion"
app_color = "blue"
app_email = "contact@tridotstech.com"
app_license = "MIT"

app_include_js = "/assets/tridots_chatbot/js/chat.js"
app_include_css = "/assets/tridots_chatbot/css/chat.css"

after_migrate = "tridots_chatbot.api.chat.after_migrate"

after_install = "tridots_chatbot.api.ingest.run_initial_ingestion"

scheduler_events = {
	"weekly": [
		"tridots_chatbot.api.ingest.run_weekly_ingestion"
	]
}

