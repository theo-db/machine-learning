import network as net
import pygame, numpy
from PIL import Image

network = net.loadNetwork("967.txt")
pygame.init()
width = 500
height = 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("number recognition")
screen.fill((0,0,0))

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill((0,0,0))
    if any(pygame.mouse.get_pressed()):
        pygame.draw.circle(screen, (255,255,255), pygame.mouse.get_pos(), 20)
    pygame.display.flip()
    st = pygame.image.tostring(screen, "RGBA", False)
    im = Image.frombytes("RGBA",(width,height),st).convert("F").resize((28,28))
    im = numpy.array(im).flatten()/256
    network.use(im)
    r = network.getResult()
    print(r)
pygame.quit()
