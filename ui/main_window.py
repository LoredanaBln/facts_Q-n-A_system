import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QScrollArea, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fact_Q_n_A_system.train.ask import load_model, load_training_data, generate_answer

class AnswerThread(QThread):
    answer_ready = pyqtSignal(str)
    
    def __init__(self, question, model, tokenizer, training_data):
        super().__init__()
        self.question = question
        self.model = model
        self.tokenizer = tokenizer
        self.training_data = training_data
        
    def run(self):
        answer = generate_answer(self.question, self.training_data)
        self.answer_ready.emit(answer)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facts Q&A System")
        self.setMinimumSize(1100, 900)
        
        try:
            self.model, self.tokenizer = load_model()
            self.training_data = load_training_data()
        except Exception as e:
            self.show_error_dialog("Model Loading Error", 
                f"Failed to load the model: {str(e)}\n\nPlease ensure the model files are present in the correct location.")
            sys.exit(1)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7f9fa;
            }
            QTextEdit {
                background-color: #fff;
                border: 1.5px solid #b0b8c1;
                border-radius: 8px;
                padding: 14px;
                font-size: 20px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 20px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QLabel {
                color: #222;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QFrame {
                background-color: #f0f4f8;
                border-radius: 8px;
                margin-top: 16px;
            }
        """)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(24)
        layout.setContentsMargins(32, 32, 32, 32)

        title = QLabel("Fact Q&A System")
        title.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        question_label = QLabel("Enter your question:")
        question_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        layout.addWidget(question_label)

        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Example: What is Earth?")
        self.question_input.setMaximumHeight(120)
        self.question_input.setFont(QFont("Segoe UI", 20))
        layout.addWidget(self.question_input)

        self.ask_button = QPushButton("Ask")
        self.ask_button.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.ask_button.setMinimumHeight(48)
        self.ask_button.clicked.connect(self.ask_question)
        layout.addWidget(self.ask_button)

        answer_label = QLabel("Answer:")
        answer_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        layout.addWidget(answer_label)

        self.answer_display = QTextEdit()
        self.answer_display.setReadOnly(True)
        self.answer_display.setFont(QFont("Segoe UI", 20))
        self.answer_display.setMinimumHeight(120)
        layout.addWidget(self.answer_display)

        topics_frame = QFrame()
        topics_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        topics_layout = QVBoxLayout(topics_frame)
        topics_layout.setSpacing(10)
        topics_layout.setContentsMargins(16, 16, 16, 16)

        topics_label = QLabel("Available Topics:")
        topics_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        topics_layout.addWidget(topics_label)

        topics_text = QTextEdit()
        topics_text.setReadOnly(True)
        topics_text.setMaximumHeight(120)
        topics_text.setFont(QFont("Segoe UI", 18))
        topics_text.setText(", ".join(sorted(self.training_data.keys())))
        topics_layout.addWidget(topics_text)

        layout.addWidget(topics_frame)
        
    def ask_question(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            return
            
        self.ask_button.setEnabled(False)
        self.answer_display.setText("Generating answer...")
        
        self.thread = AnswerThread(question, self.model, self.tokenizer, self.training_data)
        self.thread.answer_ready.connect(self.update_answer)
        self.thread.finished.connect(lambda: self.ask_button.setEnabled(True))
        self.thread.start()
        
    def update_answer(self, answer):
        self.answer_display.setText(answer)

    def show_error_dialog(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 