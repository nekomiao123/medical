def train_fn(disc_intra, disc_simu, gen_simu, gen_intra, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    
    intra_reals = 0
    intra_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (simu, intra) in enumerate(loop):
        simu = simu.to(config.DEVICE)
        intra = intra.to(config.DEVICE)

        # Train Discriminators intra and simu
        with torch.cuda.amp.autocast():
            fake_intra = gen_intra(simu)
            D_intra_real = disc_intra(intra)
            D_intra_fake = disc_intra(fake_intra.detach())
            intra_reals += D_intra_real.mean().item()
            intra_fakes += D_intra_fake.mean().item()
            D_intra_real_loss = mse(D_intra_real, torch.ones_like(D_intra_real))
            D_intra_fake_loss = mse(D_intra_fake, torch.zeros_like(D_intra_fake))
            D_intra_loss = D_intra_real_loss + D_intra_fake_loss

            fake_simu = gen_simu(intra)
            D_simu_real = disc_simu(simu)
            D_simu_fake = disc_simu(fake_simu.detach())
            D_simu_real_loss = mse(D_simu_real, torch.ones_like(D_simu_real))
            D_simu_fake_loss = mse(D_simu_fake, torch.zeros_like(D_simu_fake))
            D_simu_loss = D_simu_real_loss + D_simu_fake_loss

            # put it togethor
            D_loss = (D_intra_loss + D_simu_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators intra and simu
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_intra_fake = disc_intra(fake_intra)
            D_simu_fake = disc_simu(fake_simu)
            loss_G_intra = mse(D_intra_fake, torch.ones_like(D_intra_fake))
            loss_G_simu = mse(D_simu_fake, torch.ones_like(D_simu_fake))

            # cycle loss
            cycle_simu = gen_simu(fake_intra)
            cycle_intra = gen_intra(fake_simu)
            cycle_simu_loss = l1(simu, cycle_simu)
            cycle_intra_loss = l1(intra, cycle_intra)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_simu = gen_simu(simu)
            identity_intra = gen_intra(intra)
            identity_simu_loss = l1(simu, identity_simu)
            identity_intra_loss = l1(intra, identity_intra)

            # add all togethor
            G_loss = (
                loss_G_simu
                + loss_G_intra
                + cycle_simu_loss * config.LAMBDA_CYCLE
                + cycle_intra_loss * config.LAMBDA_CYCLE
                + identity_intra_loss * config.LAMBDA_IDENTITY
                + identity_simu_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_intra*0.5+0.5, f"saved_images/intra_{idx}.png")
            save_image(fake_simu*0.5+0.5, f"saved_images/simu_{idx}.png")

        loop.set_postfix(intra_real=intra_reals/(idx+1), intra_fake=intra_fakes/(idx+1))
