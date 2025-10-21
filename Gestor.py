import os
import logging
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import easyocr
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Cargar variables de entorno
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALLOWED_USERS = [int(id) for id in os.getenv('ALLOWED_USERS', '').split(',') if id]

class GestorCompras:
    def __init__(self):
        self.reader = easyocr.Reader(['es', 'en'], gpu=False)
        self.data_file = 'compras.json'
        self.load_data()
        logging.info("Gestor de Compras inicializado")
    
    def load_data(self):
        """Carga los datos existentes"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {'compras': []}
    
    def save_data(self):
        """Guarda los datos"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def preprocess_image(self, image_path):
        """Mejora la imagen para mejor OCR"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("No se pudo cargar la imagen")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    
    def extract_text_from_ticket(self, image_path):
        """Extrae texto de la imagen del ticket"""
        try:
            processed_image = self.preprocess_image(image_path)
            results = self.reader.readtext(processed_image, detail=1)
            full_text = ' '.join([result[1] for result in results])
            return full_text, results
        except Exception as e:
            logging.error(f"Error en OCR: {e}")
            return "", []
    
    def parse_ticket_info(self, text):
        """Extrae información específica del ticket"""
        info = {
            'fecha': None,
            'hora': None,
            'total': None,
            'subtotal': None,
            'iva': None,
            'establecimiento': None,
            'productos': []
        }
        
        # Buscar fecha (formato dd/mm/aaaa o dd-mm-aaaa)
        fecha_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        fecha_match = re.search(fecha_pattern, text)
        if fecha_match:
            info['fecha'] = fecha_match.group(1)
        
        # Buscar hora (formato hh:mm)
        hora_pattern = r'(\d{1,2}:\d{2})'
        hora_match = re.search(hora_pattern, text)
        if hora_match:
            info['hora'] = hora_match.group(1)
        
        # Buscar total (patrones comunes)
        total_patterns = [
            r'total[\s:]*\$?\s*(\d+[.,]\d+)',
            r'total[\s:]*\$?\s*(\d+)',
            r'importe[\s:]*\$?\s*(\d+[.,]\d+)',
            r'[\$€]\s*(\d+[.,]\d+)',
            r'final[\s:]*\$?\s*(\d+[.,]\d+)'
        ]
        
        for pattern in total_patterns:
            total_match = re.search(pattern, text.lower())
            if total_match:
                info['total'] = total_match.group(1).replace(',', '.')
                break
        
        # Buscar establecimiento (primeras líneas usualmente)
        lines = text.split('\n')
        if len(lines) > 0:
            # Tomar la primera línea que tenga texto significativo
            for line in lines[:3]:  # Revisar primeras 3 líneas
                if len(line.strip()) > 5 and not any(word in line.lower() for word in ['total', 'fecha', 'hora']):
                    info['establecimiento'] = line.strip()[:50]
                    break
        
        # Buscar productos (líneas con precios)
        product_pattern = r'([A-Za-z\s]+)\s+(\d+[.,]\d+)'
        for line in lines:
            product_match = re.search(product_pattern, line)
            if product_match and len(product_match.group(1).strip()) > 2:
                info['productos'].append({
                    'nombre': product_match.group(1).strip(),
                    'precio': product_match.group(2).replace(',', '.')
                })
        
        return info
    
    def add_purchase(self, ticket_info, image_path):
        """Añade una nueva compra"""
        compra = {
            'id': len(self.data['compras']) + 1,
            'fecha_compra': ticket_info.get('fecha'),
            'hora_compra': ticket_info.get('hora'),
            'establecimiento': ticket_info.get('establecimiento'),
            'total': float(ticket_info.get('total', 0)) if ticket_info.get('total') else 0,
            'productos': ticket_info.get('productos', []),
            'imagen_ticket': image_path,
            'fecha_registro': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.data['compras'].append(compra)
        self.save_data()
        return compra
    
    def get_stats(self):
        """Obtiene estadísticas de compras"""
        if not self.data['compras']:
            return None
        
        total_compras = len(self.data['compras'])
        gasto_total = sum(compra['total'] for compra in self.data['compras'])
        compra_promedio = gasto_total / total_compras if total_compras > 0 else 0
        
        # Establecimiento más frecuente
        establecimientos = [compra['establecimiento'] for compra in self.data['compras'] if compra['establecimiento']]
        establecimiento_frecuente = max(set(establecimientos), key=establecimientos.count) if establecimientos else 'N/A'
        
        stats = {
            'total_compras': total_compras,
            'gasto_total': gasto_total,
            'compra_promedio': compra_promedio,
            'establecimiento_frecuente': establecimiento_frecuente
        }
        
        return stats

class TelegramBot:
    def __init__(self, gestor_compras):
        self.gestor = gestor_compras
        self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("stats", self.stats))
        self.application.add_handler(CommandHandler("compras", self.list_compras))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Mensaje de bienvenida"""
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso a este bot.")
            return
        
        welcome_text = """
        🛒 *Gestor de Compras*
        
        Envíame una foto de tu ticket de compra y extraeré automáticamente:
        • 📅 Fecha y hora
        • 🏪 Establecimiento
        • 💰 Total de la compra
        • 🛍️ Productos (cuando sea posible)
        
        _Comandos disponibles:_
        /start - Mostrar este mensaje
        /stats - Ver estadísticas de compras
        /compras - Listar últimas compras
        """
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Procesa fotos de tickets"""
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso a este bot.")
            return
        
        await update.message.reply_text("📷 Procesando ticket...")
        
        try:
            # Descargar la foto
            photo_file = await update.message.photo[-1].get_file()
            image_path = f"tickets/ticket_{user_id}_{update.message.id}.jpg"
            
            # Crear directorio si no existe
            os.makedirs("tickets", exist_ok=True)
            
            await photo_file.download_to_drive(image_path)
            
            # Procesar con OCR
            text, raw_results = self.gestor.extract_text_from_ticket(image_path)
            ticket_info = self.gestor.parse_ticket_info(text)
            
            # Guardar en base de datos
            compra = self.gestor.add_purchase(ticket_info, image_path)
            
            # Preparar respuesta
            response = f"""
            ✅ *Ticket procesado correctamente*
            
            🏪 *Establecimiento:* {ticket_info.get('establecimiento', 'No identificado')}
            📅 *Fecha:* {ticket_info.get('fecha', 'No identificada')}
            ⏰ *Hora:* {ticket_info.get('hora', 'No identificada')}
            💰 *Total:* ${ticket_info.get('total', '0')}
            
            📋 *Productos encontrados:* {len(ticket_info['productos'])}
            """
            
            # Mostrar algunos productos si se encontraron
            if ticket_info['productos']:
                productos_text = "\n".join([f"• {p['nombre']}: ${p['precio']}" for p in ticket_info['productos'][:5]])
                response += f"\n🛍️ *Algunos productos:*\n{productos_text}"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error procesando foto: {e}")
            await update.message.reply_text("❌ Error procesando el ticket. Intenta con una foto más clara.")
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Muestra estadísticas de compras"""
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso a este bot.")
            return
        
        stats = self.gestor.get_stats()
        
        if not stats:
            await update.message.reply_text("📊 Aún no hay compras registradas.")
            return
        
        stats_text = f"""
        📊 *Estadísticas de Compras*
        
        • 🛒 Total de compras: {stats['total_compras']}
        • 💵 Gasto total: ${stats['gasto_total']:.2f}
        • 📈 Compra promedio: ${stats['compra_promedio']:.2f}
        • 🏪 Establecimiento frecuente: {stats['establecimiento_frecuente']}
        """
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def list_compras(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Lista las últimas compras"""
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso a este bot.")
            return
        
        compras = self.gestor.data['compras'][-5:]  # Últimas 5 compras
        
        if not compras:
            await update.message.reply_text("📝 No hay compras registradas.")
            return
        
        compras_text = "📝 *Últimas 5 compras:*\n\n"
        for compra in compras:
            compras_text += f"🏪 {compra['establecimiento']}\n"
            compras_text += f"📅 {compra['fecha_compra']} - 💰 ${compra['total']}\n"
            compras_text += "─" * 30 + "\n"
        
        await update.message.reply_text(compras_text, parse_mode='Markdown')
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja mensajes de texto"""
        await update.message.reply_text("📷 Envíame una foto de tu ticket para registrar la compra.\nUsa /stats para ver estadísticas o /compras para listar compras.")
    
    def run(self):
        """Inicia el bot"""
        self.application.run_polling()

def main():
    logging.info("Iniciando Gestor de Compras...")
    
    # Inicializar gestor
    gestor = GestorCompras()
    
    # Inicializar y ejecutar bot
    bot = TelegramBot(gestor)
    bot.run()

if __name__ == "__main__":
    main()
