import subprocess
import sys

try:
    import os
    import json
    import customtkinter
    from pathlib import Path
    import codecs
except ImportError:
    print('Не все библиотеки установлены.')
    os.system('pip install customtkinter pathlib')

# Открытие сохраненных данных
with codecs.open(Path('files/config.json').resolve(), 'r', 'utf-8') as data:
    config = json.load(data)
    data.close()

# Конфигурация интерфейса
if config['theme2'] == 'светлая':
    customtkinter.set_appearance_mode('light')
    colorback = '#F2F2F2'

elif config['theme2'] == 'тёмная':
    customtkinter.set_appearance_mode('dark')
    colorback = '#1A1A1A'

if config['theme'] == 'оранжевый':
    color1 = '#F07427'
    color2 = '#DF5900'

elif config['theme'] == 'зелёный':
    color1 = '#55A376'
    color2 = '#306846'

elif config['theme'] == 'синий':
    color1 = '#3669A0'
    color2 = '#273B4D'

elif config['theme'] == 'красный':
    color1 = '#EB4C42'
    color2 = '#CD443A'

elif config['theme'] == 'бирюзовый':
    color1 = '#1CC3BB'
    color2 = '#19AEA7'

customtkinter.set_default_color_theme('dark-blue')
app = customtkinter.CTk()
app.geometry('590x370')
app.title('SmartControl')
app.resizable(width=False, height=False)
app.after(201, lambda: app.iconbitmap(Path('files/losmart.ico').resolve()))


def save():
    config['wakeword'] = wakeword_entry.get().lower()
    config['voice'] = voice_entry.get().lower()
    config['time'] = int(time_entry.get().lower())
    config['theme'] = theme_entry.get().lower()
    config['theme2'] = theme2_entry.get().lower()
    config['rasp_type'] = rasp_type_entry.get()
    config['rasp'] = vosk_entry.get().lower()
    config['sintez'] = silero_entry.get().lower()
    config['gc_api'] = gc_api_entry.get()
    with codecs.open(Path('files/config.json').resolve(), 'w', 'utf-8') as config_file:
        json.dump(config, config_file, ensure_ascii=False)
        config_file.close()


def start():
    save()
    subprocess.run([sys.executable, 'SmartControl+.py'])


def doc():
    os.startfile(Path('files/doc.pdf').resolve())


def add():
    subprocess.run([sys.executable, "configurator.pyw"])


# Разметка интерфейса
tabview = customtkinter.CTkTabview(master=app, fg_color=colorback, segmented_button_selected_hover_color=color1,
                                   segmented_button_selected_color=color1,
                                   segmented_button_unselected_hover_color=color2)
tabview.place(relx=0.5, rely=0.35, anchor=customtkinter.CENTER)

tabview.add('Настройки ассистента')
tabview.add('Конфигурация ассистента')
tabview.add('Настройки приложения')
tabview.add('       API       ')

label_settings_assistant = customtkinter.CTkLabel(tabview.tab('Настройки ассистента'), text='Настройки ассистента',
                                                  bg_color=colorback, font=('TkHeadingFont', 15.1))
label_settings_assistant.place(relx=0.05, rely=0.1, anchor=customtkinter.W)

label_config_assistant = customtkinter.CTkLabel(tabview.tab('Конфигурация ассистента'), text='Конфигурация ассистента',
                                                bg_color=colorback, font=('TkHeadingFont', 15.1))
label_config_assistant.place(relx=0.05, rely=0.1, anchor=customtkinter.W)

label_settings_app = customtkinter.CTkLabel(tabview.tab('Настройки приложения'), text='Настройки приложения',
                                            bg_color=colorback, font=('TkHeadingFont', 15.1))
label_settings_app.place(relx=0.05, rely=0.1, anchor=customtkinter.W)

label_api = customtkinter.CTkLabel(tabview.tab('       API       '), text='API', bg_color=colorback,
                                   font=('TkHeadingFont', 15.1))
label_api.place(relx=0.05, rely=0.1, anchor=customtkinter.W)

label_wakeword = customtkinter.CTkLabel(tabview.tab('Настройки ассистента'), text='Активационная фраза:',
                                        bg_color=colorback, font=('TkHeadingFont', 14))
label_wakeword.place(relx=0.05, rely=0.3, anchor=customtkinter.W)

wakeword_entry = customtkinter.CTkComboBox(tabview.tab('Настройки ассистента'), width=230, values=[config['wakeword']],
                                           border_color=color1, button_color=color1, button_hover_color=color2,
                                           corner_radius=5, font=('TkHeadingFont', 14))
wakeword_entry.set(config['wakeword'])
wakeword_entry.place(relx=0.95, rely=0.3, anchor=customtkinter.E)

label_voice = customtkinter.CTkLabel(tabview.tab('Настройки ассистента'), text='Голос:', bg_color=colorback,
                                     font=('TkHeadingFont', 14))
label_voice.place(relx=0.05, rely=0.6, anchor=customtkinter.W)

voice_entry = customtkinter.CTkOptionMenu(tabview.tab('Настройки ассистента'), width=230,
                                          values=['xenia', 'kseniya', 'baya'], fg_color=color1, button_color=color1,
                                          button_hover_color=color2, corner_radius=5, font=('TkHeadingFont', 14))
voice_entry.set(config['voice'])
voice_entry.place(relx=0.95, rely=0.6, anchor=customtkinter.E)

label_time = customtkinter.CTkLabel(tabview.tab('Настройки ассистента'), text='Приём команды без акт. фразы (с):',
                                    bg_color=colorback, font=('TkHeadingFont', 14))
label_time.place(relx=0.05, rely=0.9, anchor=customtkinter.W)

time_entry = customtkinter.CTkComboBox(tabview.tab('Настройки ассистента'), width=230,
                                       values=list(map(str, [config['time']])), border_color=color1,
                                       button_color=color1, button_hover_color=color2, corner_radius=5,
                                       font=('TkHeadingFont', 14))
time_entry.set(str(config['time']))
time_entry.place(relx=0.95, rely=0.9, anchor=customtkinter.E)

label_rasp_type = customtkinter.CTkLabel(tabview.tab('Конфигурация ассистента'), text='Вариант распознавания:',
                                         bg_color=colorback, font=('TkHeadingFont', 14))
label_rasp_type.place(relx=0.05, rely=0.3, anchor=customtkinter.W)

rasp_type_entry = customtkinter.CTkOptionMenu(tabview.tab('Конфигурация ассистента'), width=230,
                                              values=['Vosk', 'Google Speech Recognition'], fg_color=color1,
                                              button_color=color1, button_hover_color=color2, corner_radius=5,
                                              font=('TkHeadingFont', 14))
rasp_type_entry.set(config['rasp_type'])
rasp_type_entry.place(relx=0.95, rely=0.3, anchor=customtkinter.E)

label_vosk = customtkinter.CTkLabel(tabview.tab('Конфигурация ассистента'), text='Версия модели Vosk: ',
                                    bg_color=colorback, font=('TkHeadingFont', 14))
label_vosk.place(relx=0.05, rely=0.6, anchor=customtkinter.W)

vosk_entry = customtkinter.CTkOptionMenu(tabview.tab('Конфигурация ассистента'), width=230, values=['0.22', '0.4'],
                                         fg_color=color1, button_color=color1, button_hover_color=color2,
                                         corner_radius=5, font=('TkHeadingFont', 14))
vosk_entry.set(config['rasp'])
vosk_entry.place(relx=0.95, rely=0.6, anchor=customtkinter.E)

label_silero = customtkinter.CTkLabel(tabview.tab('Конфигурация ассистента'), text='Синтез речи:', bg_color=colorback,
                                      font=('TkHeadingFont', 14))
label_silero.place(relx=0.05, rely=0.9, anchor=customtkinter.W)

silero_entry = customtkinter.CTkOptionMenu(tabview.tab('Конфигурация ассистента'), width=230,
                                           values=['ru_v3', 'v3_1_ru'], fg_color=color1, button_color=color1,
                                           button_hover_color=color2, corner_radius=5, font=('TkHeadingFont', 14))
silero_entry.set(config['sintez'])
silero_entry.place(relx=0.95, rely=0.9, anchor=customtkinter.E)

label_theme2 = customtkinter.CTkLabel(tabview.tab('Настройки приложения'), text='Тема:', bg_color=colorback,
                                      font=('TkHeadingFont', 14))
label_theme2.place(relx=0.05, rely=0.3, anchor=customtkinter.W)

theme2_entry = customtkinter.CTkOptionMenu(tabview.tab('Настройки приложения'), width=230, values=['светлая', 'тёмная'],
                                           fg_color=color1, button_color=color1, button_hover_color=color2,
                                           corner_radius=5, font=('TkHeadingFont', 14))
theme2_entry.set(config['theme2'])
theme2_entry.place(relx=0.95, rely=0.3, anchor=customtkinter.E)

label_theme = customtkinter.CTkLabel(tabview.tab('Настройки приложения'), text='Акцентный цвет:', bg_color=colorback,
                                     font=('TkHeadingFont', 14))
label_theme.place(relx=0.05, rely=0.6, anchor=customtkinter.W)

theme_entry = customtkinter.CTkOptionMenu(tabview.tab('Настройки приложения'), width=230,
                                          values=['оранжевый', 'зелёный', 'синий', 'красный', 'бирюзовый'],
                                          fg_color=color1, button_color=color1, button_hover_color=color2,
                                          corner_radius=5, font=('TkHeadingFont', 14))
theme_entry.set(config['theme'])
theme_entry.place(relx=0.95, rely=0.6, anchor=customtkinter.E)

label_gc_api = customtkinter.CTkLabel(tabview.tab('       API       '), text='GigaChat:', bg_color=colorback,
                                      font=('TkHeadingFont', 14))
label_gc_api.place(relx=0.05, rely=0.3, anchor=customtkinter.W)

gc_api_entry = customtkinter.CTkComboBox(tabview.tab('       API       '), width=230, values=[config['gc_api']],
                                         border_color=color1, button_color=color1, button_hover_color=color2,
                                         corner_radius=5, font=('TkHeadingFont', 14))
gc_api_entry.set(config['gc_api'])
gc_api_entry.place(relx=0.95, rely=0.3, anchor=customtkinter.E)

button_1 = customtkinter.CTkButton(master=app, text='Сохранить', width=230, fg_color=color1, hover_color=color2,
                                   font=('TkHeadingFont', 15), corner_radius=5, command=save)
button_1.place(relx=0.07, rely=0.765, anchor=customtkinter.W)

button_2 = customtkinter.CTkButton(master=app, text='Запустить', width=230, fg_color=color1, hover_color=color2,
                                   font=('TkHeadingFont', 15), corner_radius=5, command=start)
button_2.place(relx=0.93, rely=0.765, anchor=customtkinter.E)

button_3 = customtkinter.CTkButton(master=app, text='Команды', width=230, fg_color=color1, hover_color=color2,
                                   font=('TkHeadingFont', 15), corner_radius=5, command=add)
button_3.place(relx=0.07, rely=0.9, anchor=customtkinter.W)

button_4 = customtkinter.CTkButton(master=app, text='Справка', width=230, fg_color=color1, hover_color=color2,
                                   font=('TkHeadingFont', 15), corner_radius=5, command=doc)
button_4.place(relx=0.93, rely=0.9, anchor=customtkinter.E)

# Запуск интерфейса
app.mainloop()
